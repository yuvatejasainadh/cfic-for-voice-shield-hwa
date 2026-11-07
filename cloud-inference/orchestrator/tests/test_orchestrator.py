"""Tests for Cloud Inference Orchestrator."""

import io
from unittest.mock import AsyncMock, patch
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.routing.language_router import language_router
from app.routing.spoof_selector import spoof_selector


def test_language_router_selection():
    # English routes to English ASR
    url, lang = language_router.select_asr_endpoint("en")
    assert lang == "en"
    assert url is not None

    # Telugu routes to Telugu IndicConformer
    url, lang = language_router.select_asr_endpoint("te")
    assert lang == "te"
    assert url is not None

    # Hindi routes to Hindi IndicConformer
    url, lang = language_router.select_asr_endpoint("hi")
    assert lang == "hi"
    assert url is not None

    # Unsupported languages return None for ASR endpoint (no silent substitution)
    for unsupported in ["ta", "kn", "ml", "mr", "bn", "gu", "pa", "as", "mni", "fr", "de", "es", "zh", "unknown"]:
        url, lang = language_router.select_asr_endpoint(unsupported)
        assert url is None
        assert lang == unsupported


def test_spoof_selector():
    # South Indian languages select VMC-South
    for lang in ["te", "ta", "kn", "ml"]:
        url, model = spoof_selector.select_spoof_endpoint(lang)
        assert model == "vmc-south"

    # English and general fallback select AASIST-L fallback
    for lang in ["en", "hi", "mr", "unknown"]:
        url, model = spoof_selector.select_spoof_endpoint(lang)
        assert model == "aasist-fallback"


@pytest.mark.asyncio
async def test_orchestrator_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["service"] == "cloud-inference-orchestrator"
        assert "lid" in data["backends"]
        assert "asr_en" in data["backends"]
        assert "vmc_south" in data["backends"]
        assert "rias" in data["backends"]


@pytest.mark.asyncio
async def test_inference_pipeline_english_success():
    mock_lid = {
        "status": "success",
        "language": "en",
        "raw_label": "en-IN",
        "confidence": 0.98,
        "probabilities": {"en": 0.98, "hi": 0.01},
        "top_k": [{"language": "en", "raw_label": "en-IN", "confidence": 0.98}],
        "model_id": "onecxi/open-vakgyata",
        "model_revision": "f2754058e485dfc65cc62589b8d0e21c1d328399",
        "retries_used": 0,
        "latency_ms": 120.0,
    }
    mock_asr = {
        "status": "success",
        "text": "The studies indicate that there is some utility.",
        "language": "en",
        "model_id": "openai/whisper-base.en",
        "model_revision": "911407f4214e0e1d82085af863093ec0b66f9cd6",
        "latency_ms": 350.0,
    }
    mock_spoof = {
        "status": "success",
        "prediction": "GENUINE",
        "score": 0.05,
        "model_version": "aasist-fallback",
        "latency_ms": 150.0,
    }
    mock_rias = {
        "status": "success",
        "risk_level": "LOW",
        "failure_reason": None,
        "model_id": "google/gemma-4-E2B-it",
        "model_revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
        "latency_ms": 210.0,
    }

    with patch("app.api.routes.lid_client.identify", new=AsyncMock(return_value=mock_lid)), \
         patch("app.api.routes.asr_client.transcribe", new=AsyncMock(return_value=mock_asr)), \
         patch("app.api.routes.spoof_client.predict", new=AsyncMock(return_value=mock_spoof)), \
         patch("app.api.routes.rias_client.assess", new=AsyncMock(return_value=mock_rias)):

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            files = {"file": ("chunk.wav", b"dummy-audio-content", "audio/wav")}
            data = {"session_id": "sess-123", "chunk_index": "2", "timestamp_start_s": "20.0", "timestamp_end_s": "30.0"}
            res = await ac.post("/v1/inference", files=files, data=data)

            assert res.status_code == 200
            res_data = res.json()
            assert res_data["status"] == "success"
            assert res_data["session_id"] == "sess-123"
            assert res_data["chunk_index"] == 2
            assert res_data["lid"]["language"] == "en"
            assert res_data["lid"]["confidence"] == 0.98
            assert res_data["transcription"]["status"] == "success"
            assert res_data["transcription"]["text"] == "The studies indicate that there is some utility."
            assert res_data["spoof"]["prediction"] == "GENUINE"
            assert res_data["rias"] is not None
            assert res_data["rias"]["status"] == "success"
            assert res_data["rias"]["risk_level"] == "LOW"


@pytest.mark.asyncio
async def test_inference_pipeline_high_risk_rias():
    mock_lid = {
        "status": "success",
        "language": "en",
        "raw_label": "en-IN",
        "confidence": 0.95,
        "model_id": "onecxi/open-vakgyata",
        "retries_used": 0,
        "latency_ms": 100.0,
    }
    mock_asr = {
        "status": "success",
        "text": "This is police headquarters. You are under digital arrest. Transfer funds now.",
        "language": "en",
        "model_id": "openai/whisper-base.en",
        "latency_ms": 300.0,
    }
    mock_spoof = {
        "status": "success",
        "prediction": "GENUINE",
        "score": 0.02,
        "model_version": "aasist-fallback",
        "latency_ms": 120.0,
    }
    mock_rias = {
        "status": "success",
        "risk_level": "HIGH",
        "failure_reason": None,
        "model_id": "google/gemma-4-E2B-it",
        "model_revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
        "latency_ms": 180.0,
    }

    with patch("app.api.routes.lid_client.identify", new=AsyncMock(return_value=mock_lid)), \
         patch("app.api.routes.asr_client.transcribe", new=AsyncMock(return_value=mock_asr)), \
         patch("app.api.routes.spoof_client.predict", new=AsyncMock(return_value=mock_spoof)), \
         patch("app.api.routes.rias_client.assess", new=AsyncMock(return_value=mock_rias)):

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            files = {"file": ("chunk.wav", b"dummy-audio-content", "audio/wav")}
            res = await ac.post("/v1/inference", files=files)

            assert res.status_code == 200
            res_data = res.json()
            assert res_data["status"] == "success"
            assert res_data["rias"]["status"] == "success"
            assert res_data["rias"]["risk_level"] == "HIGH"


@pytest.mark.asyncio
async def test_inference_pipeline_low_confidence_lid_skips_asr():
    mock_lid = {
        "status": "success",
        "language": "en",
        "raw_label": "en-IN",
        "confidence": 0.45,
        "probabilities": {"en": 0.45, "hi": 0.40},
        "model_id": "onecxi/open-vakgyata",
        "retries_used": 2,
        "latency_ms": 150.0,
    }
    mock_spoof = {
        "status": "success",
        "prediction": "GENUINE",
        "score": 0.05,
        "model_version": "aasist-fallback",
        "latency_ms": 100.0,
    }

    mock_asr_call = AsyncMock()
    mock_rias_call = AsyncMock()

    with patch("app.api.routes.lid_client.identify", new=AsyncMock(return_value=mock_lid)), \
         patch("app.api.routes.asr_client.transcribe", new=mock_asr_call), \
         patch("app.api.routes.spoof_client.predict", new=AsyncMock(return_value=mock_spoof)), \
         patch("app.api.routes.rias_client.assess", new=mock_rias_call):

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            files = {"file": ("chunk.wav", b"dummy-audio-content", "audio/wav")}
            res = await ac.post("/v1/inference", files=files)

            assert res.status_code == 200
            res_data = res.json()
            assert res_data["status"] == "inconclusive"
            assert res_data["lid"]["status"] == "inconclusive"
            assert res_data["transcription"]["status"] == "skipped"
            assert res_data["transcription"]["text"] is None
            assert res_data["rias"]["status"] == "skipped"
            assert res_data["rias"]["risk_level"] == "INCONCLUSIVE"
            # Verify ASR and RIAS were NOT called
            mock_asr_call.assert_not_called()
            mock_rias_call.assert_not_called()


@pytest.mark.asyncio
async def test_inference_pipeline_unsupported_language_skips_asr_and_rias():
    mock_lid = {
        "status": "success",
        "language": "as",
        "raw_label": "as-IN",
        "confidence": 0.98,
        "model_id": "onecxi/open-vakgyata",
        "retries_used": 0,
        "latency_ms": 110.0,
    }
    mock_spoof = {
        "status": "success",
        "prediction": "GENUINE",
        "score": 0.05,
        "model_version": "aasist-fallback",
        "latency_ms": 100.0,
    }
    mock_asr_call = AsyncMock()
    mock_rias_call = AsyncMock()

    with patch("app.api.routes.lid_client.identify", new=AsyncMock(return_value=mock_lid)), \
         patch("app.api.routes.asr_client.transcribe", new=mock_asr_call), \
         patch("app.api.routes.spoof_client.predict", new=AsyncMock(return_value=mock_spoof)), \
         patch("app.api.routes.rias_client.assess", new=mock_rias_call):

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            files = {"file": ("chunk.wav", b"dummy-audio-content", "audio/wav")}
            res = await ac.post("/v1/inference", files=files)

            assert res.status_code == 200
            res_data = res.json()
            assert res_data["status"] == "inconclusive"
            assert res_data["lid"]["language"] == "as"
            assert res_data["transcription"]["status"] == "unsupported_language"
            assert res_data["transcription"]["error"] == "Language 'as' is not supported by ASR services"
            assert res_data["rias"]["status"] == "skipped"
            assert res_data["rias"]["risk_level"] == "INCONCLUSIVE"
            mock_asr_call.assert_not_called()
            mock_rias_call.assert_not_called()


@pytest.mark.asyncio
async def test_inference_pipeline_asr_failure_inconclusive():
    mock_lid = {
        "status": "success",
        "language": "en",
        "raw_label": "en-IN",
        "confidence": 0.95,
        "model_id": "onecxi/open-vakgyata",
        "retries_used": 0,
        "latency_ms": 100.0,
    }
    mock_asr = {
        "status": "error",
        "text": None,
        "language": "en",
        "error": "ASR downstream timeout",
        "latency_ms": 5000.0,
    }
    mock_spoof = {
        "status": "success",
        "prediction": "GENUINE",
        "score": 0.05,
        "model_version": "aasist-fallback",
        "latency_ms": 100.0,
    }
    mock_rias_call = AsyncMock()

    with patch("app.api.routes.lid_client.identify", new=AsyncMock(return_value=mock_lid)), \
         patch("app.api.routes.asr_client.transcribe", new=AsyncMock(return_value=mock_asr)), \
         patch("app.api.routes.spoof_client.predict", new=AsyncMock(return_value=mock_spoof)), \
         patch("app.api.routes.rias_client.assess", new=mock_rias_call):

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            files = {"file": ("chunk.wav", b"dummy-audio-content", "audio/wav")}
            res = await ac.post("/v1/inference", files=files)

            assert res.status_code == 200
            res_data = res.json()
            assert res_data["status"] == "inconclusive"
            assert res_data["transcription"]["status"] == "error"
            assert res_data["rias"]["status"] == "skipped"
            assert res_data["rias"]["risk_level"] == "INCONCLUSIVE"
            mock_rias_call.assert_not_called()


@pytest.mark.asyncio
async def test_inference_pipeline_rias_failure_inconclusive():
    mock_lid = {
        "status": "success",
        "language": "en",
        "confidence": 0.95,
    }
    mock_asr = {
        "status": "success",
        "text": "Hello, how are you today?",
        "language": "en",
    }
    mock_spoof = {
        "status": "success",
        "prediction": "GENUINE",
    }
    mock_rias = {
        "status": "inconclusive",
        "risk_level": "INCONCLUSIVE",
        "failure_reason": "INFERENCE_TIMEOUT",
    }

    with patch("app.api.routes.lid_client.identify", new=AsyncMock(return_value=mock_lid)), \
         patch("app.api.routes.asr_client.transcribe", new=AsyncMock(return_value=mock_asr)), \
         patch("app.api.routes.spoof_client.predict", new=AsyncMock(return_value=mock_spoof)), \
         patch("app.api.routes.rias_client.assess", new=AsyncMock(return_value=mock_rias)):

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            files = {"file": ("chunk.wav", b"dummy-audio-content", "audio/wav")}
            res = await ac.post("/v1/inference", files=files)

            assert res.status_code == 200
            res_data = res.json()
            assert res_data["status"] == "inconclusive"
            assert res_data["rias"]["status"] == "inconclusive"
            assert res_data["rias"]["risk_level"] == "INCONCLUSIVE"
            assert res_data["rias"]["failure_reason"] == "INFERENCE_TIMEOUT"
