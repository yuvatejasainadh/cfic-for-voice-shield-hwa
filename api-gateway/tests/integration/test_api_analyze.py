"""Integration tests for the /v1/analyze audio analysis pipeline."""

import io
import wave
import pytest
from httpx import ASGITransport, AsyncClient
from app.api.routes import rate_limiter
from app.core.config import settings
from app.main import app
from services.pella.client import PellaClient
from services.transcription.client import TranscriptionClient


def make_test_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a minimal valid PCM 16-bit mono WAV in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(sample_rate * duration_s))
    return buf.getvalue()


@pytest.fixture(autouse=True)
def reset_limiter():
    rate_limiter.reset()


@pytest.mark.asyncio
async def test_analyze_pipeline_mandatory_high_override(monkeypatch):
    """Verify that when RIAS reports HIGH risk in legacy mode, final outcome is SUSPICIOUS."""
    monkeypatch.setattr(settings, "use_cloud_inference", False)

    async def mock_pella_predict(self, audio, filename, content_type):
        return {
            "status": "success",
            "prediction": "GENUINE",
            "score": 0.05,
            "model_version": "pellav2-mock-v1",
            "latency_ms": 25.0,
        }

    async def mock_transcribe(self, audio, filename, content_type):
        return {
            "status": "success",
            "text": "Please provide your debit card CVV immediately or police will arrive",
            "language": "en",
            "model_version": "whisper-mock-v1",
            "latency_ms": 30.0,
        }

    async def mock_assess_transcript(transcript, **kwargs):
        return {
            "status": "success",
            "risk_level": "HIGH",
            "reason": "Extortion and coercive bank credential demand",
            "latency_ms": 40.0,
        }

    monkeypatch.setattr(PellaClient, "predict", mock_pella_predict)
    monkeypatch.setattr(TranscriptionClient, "transcribe", mock_transcribe)
    monkeypatch.setattr("app.orchestration.service.assess_transcript", mock_assess_transcript)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("call.wav", make_test_wav(1.0), "audio/wav")}
        headers = {"X-API-Key": settings.api_key}
        response = await ac.post("/v1/analyze", files=files, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["final_outcome"] == "SUSPICIOUS"
        assert data["reason_code"] == "HIGH_RIAS_OVERRIDE"
        assert data["pella_status"] == "success"
        assert data["rias_transcription_status"] == "success"
        assert data["rias_risk_status"] == "success"
        assert data["rias_risk_level"] == "HIGH"
        assert "total" in data["latency_ms"]
        assert "pella" in data["latency_ms"]
        assert "transcription" in data["latency_ms"]


@pytest.mark.asyncio
async def test_analyze_pipeline_downstream_timeout_handled_safely(monkeypatch):
    """Verify that downstream timeout does not crash the API, returns INCONCLUSIVE in legacy mode."""
    monkeypatch.setattr(settings, "use_cloud_inference", False)

    async def mock_pella_timeout(self, audio, filename, content_type):
        return {
            "status": "timeout",
            "prediction": None,
            "score": None,
            "model_version": None,
            "error": "Upstream timeout",
            "latency_ms": 1000.0,
        }

    async def mock_transcribe_timeout(self, audio, filename, content_type):
        return {
            "status": "timeout",
            "text": None,
            "language": None,
            "model_version": None,
            "error": "Upstream timeout",
            "latency_ms": 1000.0,
        }

    monkeypatch.setattr(PellaClient, "predict", mock_pella_timeout)
    monkeypatch.setattr(TranscriptionClient, "transcribe", mock_transcribe_timeout)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("call.wav", make_test_wav(1.0), "audio/wav")}
        headers = {"X-API-Key": settings.api_key}
        response = await ac.post("/v1/analyze", files=files, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["final_outcome"] == "INCONCLUSIVE"
        assert data["reason_code"] == "INSUFFICIENT_EVIDENCE"
        assert data["pella_status"] == "timeout"
        assert data["rias_transcription_status"] == "timeout"


@pytest.mark.asyncio
async def test_cloud_inference_completes_past_30s(monkeypatch):
    """Verify that Cloud Inference completing in 35-40s is successfully aggregated without 30s timeout."""
    monkeypatch.setattr(settings, "use_cloud_inference", True)
    monkeypatch.setattr(settings, "cloud_inference_timeout", 60.0)
    monkeypatch.setattr(settings, "allow_legacy_fallback", False)

    async def mock_cloud_run_inference(self, audio, filename, content_type, **kwargs):
        # Simulates successful cloud inference completing at 36.5 seconds
        return {
            "status": "success",
            "chunk_index": kwargs.get("chunk_index", 0),
            "spoof": {
                "status": "success",
                "prediction": "GENUINE",
                "score": 0.045,
                "model_version": "pellav2-original-fp32",
                "latency_ms": 95.0,
            },
            "transcription": {
                "status": "success",
                "text": "Hello, this is standard customer service verification.",
                "language": "en",
                "model_id": "openai/whisper-base.en",
                "latency_ms": 7500.0,
            },
            "lid": {"status": "success", "language": "en", "confidence": 0.98},
            "rias": {
                "status": "success",
                "risk_level": "LOW",
                "failure_reason": None,
                "model_id": "google/gemma-4-E2B-it",
                "latency_ms": 26000.0,
            },
            "latency_ms": 36500.0,
        }

    from app.services.cfic_client import CFICInferenceClient
    monkeypatch.setattr(CFICInferenceClient, "run_inference", mock_cloud_run_inference)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("call.wav", make_test_wav(1.0), "audio/wav")}
        headers = {"X-API-Key": settings.api_key}
        response = await ac.post("/v1/analyze", files=files, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["final_outcome"] == "GENUINE"
        assert data["reason_code"] == "ACOUSTIC_GENUINE_CONFIRMED"
        assert data["pella_status"] == "success"
        assert data["rias_transcription_status"] == "success"
        assert data["rias_risk_status"] == "success"
        assert data["rias_risk_level"] == "LOW"
        assert data["model_version"] == "pellav2-original-fp32"


@pytest.mark.asyncio
async def test_cloud_inference_timeout_handling_in_cloud_mode(monkeypatch):
    """Verify that Cloud Inference timeout returns controlled INCONCLUSIVE without calling dead localhost."""
    monkeypatch.setattr(settings, "use_cloud_inference", True)
    monkeypatch.setattr(settings, "cloud_inference_timeout", 60.0)
    monkeypatch.setattr(settings, "allow_legacy_fallback", False)

    async def mock_cloud_timeout(self, audio, filename, content_type, **kwargs):
        return {
            "status": "timeout",
            "error": "Cloud inference request timed out",
            "spoof": {"status": "timeout", "prediction": None, "score": None, "latency_ms": 60000.0},
            "transcription": {"status": "timeout", "text": None, "language": None, "latency_ms": 60000.0},
            "lid": {"status": "timeout", "language": None},
            "rias": {"status": "timeout", "risk_level": "INCONCLUSIVE", "failure_reason": "Cloud inference request timed out", "latency_ms": 60000.0},
            "latency_ms": 60000.0,
        }

    def dead_legacy_called(*args, **kwargs):
        raise AssertionError("Dead legacy localhost client must NOT be called in cloud mode!")

    from app.services.cfic_client import CFICInferenceClient
    monkeypatch.setattr(CFICInferenceClient, "run_inference", mock_cloud_timeout)
    monkeypatch.setattr(PellaClient, "predict", dead_legacy_called)
    monkeypatch.setattr(TranscriptionClient, "transcribe", dead_legacy_called)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("call.wav", make_test_wav(1.0), "audio/wav")}
        headers = {"X-API-Key": settings.api_key}
        response = await ac.post("/v1/analyze", files=files, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["final_outcome"] == "INCONCLUSIVE"
        assert data["reason_code"] == "INSUFFICIENT_EVIDENCE"
        assert data["pella_status"] == "timeout"
        assert data["rias_transcription_status"] == "timeout"
        assert data["rias_risk_status"] == "timeout"


@pytest.mark.asyncio
async def test_cloud_inference_connection_failure_preserves_error(monkeypatch):
    """Verify that cloud connection failure preserves error category without calling dead localhost."""
    monkeypatch.setattr(settings, "use_cloud_inference", True)
    monkeypatch.setattr(settings, "allow_legacy_fallback", False)

    async def mock_cloud_conn_error(self, audio, filename, content_type, **kwargs):
        return {
            "status": "error",
            "error": "Cloud inference request failed: ConnectError",
            "spoof": {"status": "error", "prediction": None, "score": None, "latency_ms": 50.0},
            "transcription": {"status": "error", "text": None, "language": None, "latency_ms": 50.0},
            "lid": {"status": "error", "language": None},
            "rias": {"status": "error", "risk_level": "INCONCLUSIVE", "failure_reason": "ConnectError", "latency_ms": 50.0},
            "latency_ms": 50.0,
        }

    def dead_legacy_called(*args, **kwargs):
        raise AssertionError("Dead legacy localhost client must NOT be called in cloud mode!")

    from app.services.cfic_client import CFICInferenceClient
    monkeypatch.setattr(CFICInferenceClient, "run_inference", mock_cloud_conn_error)
    monkeypatch.setattr(PellaClient, "predict", dead_legacy_called)
    monkeypatch.setattr(TranscriptionClient, "transcribe", dead_legacy_called)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("call.wav", make_test_wav(1.0), "audio/wav")}
        headers = {"X-API-Key": settings.api_key}
        response = await ac.post("/v1/analyze", files=files, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["final_outcome"] == "INCONCLUSIVE"
        assert data["reason_code"] == "INSUFFICIENT_EVIDENCE"
        assert data["pella_status"] == "error"
        assert data["rias_transcription_status"] == "error"
        assert data["rias_risk_status"] == "error"


@pytest.mark.asyncio
async def test_legacy_fallback_when_explicitly_enabled(monkeypatch):
    """Verify that fallback to legacy client occurs ONLY when allow_legacy_fallback is explicitly enabled."""
    monkeypatch.setattr(settings, "use_cloud_inference", True)
    monkeypatch.setattr(settings, "allow_legacy_fallback", True)

    async def mock_cloud_error(self, audio, filename, content_type, **kwargs):
        return {
            "status": "error",
            "error": "Cloud inference down",
            "spoof": {"status": "error", "prediction": None, "score": None, "latency_ms": 10.0},
            "transcription": {"status": "error", "text": None, "language": None, "latency_ms": 10.0},
            "lid": {"status": "error", "language": None},
            "rias": {"status": "error", "risk_level": "INCONCLUSIVE", "latency_ms": 10.0},
            "latency_ms": 10.0,
        }

    async def mock_pella_legacy(self, audio, filename, content_type):
        return {
            "status": "success",
            "prediction": "FAKE",
            "score": 0.99,
            "model_version": "pellav2-fallback",
            "latency_ms": 20.0,
        }

    async def mock_transcribe_legacy(self, audio, filename, content_type):
        return {
            "status": "success",
            "text": "Fallback text",
            "language": "en",
            "model_version": "whisper-fallback",
            "latency_ms": 25.0,
        }

    from app.services.cfic_client import CFICInferenceClient
    monkeypatch.setattr(CFICInferenceClient, "run_inference", mock_cloud_error)
    monkeypatch.setattr(PellaClient, "predict", mock_pella_legacy)
    monkeypatch.setattr(TranscriptionClient, "transcribe", mock_transcribe_legacy)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("call.wav", make_test_wav(1.0), "audio/wav")}
        headers = {"X-API-Key": settings.api_key}
        response = await ac.post("/v1/analyze", files=files, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["final_outcome"] == "FAKE"
        assert data["reason_code"] == "ACOUSTIC_SPOOF_DETECTED"
        assert data["pella_status"] == "success"
        assert data["model_version"] == "pellav2-fallback"
