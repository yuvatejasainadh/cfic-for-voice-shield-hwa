"""Integration tests for the /v1/analyze audio analysis pipeline."""

import pytest
from httpx import ASGITransport, AsyncClient
from app.api.routes import rate_limiter
from app.core.config import settings
from app.main import app
from services.pella.client import PellaClient
from services.transcription.client import TranscriptionClient


@pytest.fixture(autouse=True)
def reset_limiter():
    rate_limiter.reset()


@pytest.mark.asyncio
async def test_analyze_pipeline_mandatory_high_override(monkeypatch):
    """Verify that when RIAS reports HIGH risk, final outcome is SUSPICIOUS."""

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
        files = {"file": ("call.wav", b"fake audio bytes for testing", "audio/wav")}
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
    """Verify that downstream timeout does not crash the API, returns INCONCLUSIVE."""

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
        files = {"file": ("call.wav", b"sample audio bytes", "audio/wav")}
        headers = {"X-API-Key": settings.api_key}
        response = await ac.post("/v1/analyze", files=files, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["final_outcome"] == "INCONCLUSIVE"
        assert data["reason_code"] == "INSUFFICIENT_EVIDENCE"
        assert data["pella_status"] == "timeout"
        assert data["rias_transcription_status"] == "timeout"
