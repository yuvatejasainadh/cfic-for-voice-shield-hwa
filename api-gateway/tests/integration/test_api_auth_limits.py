"""Integration tests for authentication, payload limits, and rate limiting."""

import pytest
from httpx import ASGITransport, AsyncClient
from app.api.routes import rate_limiter
from app.core.config import settings
from app.main import app
from app.schemas import AnalysisResponse, FinalOutcome


@pytest.fixture(autouse=True)
def reset_limiter():
    rate_limiter.reset()


@pytest.mark.asyncio
async def test_missing_api_key_returns_401():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("audio.wav", b"dummy audio content", "audio/wav")}
        response = await ac.post("/v1/analyze", files=files)
        assert response.status_code == 401
        data = response.json()
        assert data["error_code"] == "UNAUTHORIZED"
        assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_invalid_api_key_returns_401():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("audio.wav", b"dummy audio content", "audio/wav")}
        headers = {"X-API-Key": "wrong-key-12345"}
        response = await ac.post("/v1/analyze", files=files, headers=headers)
        assert response.status_code == 401
        data = response.json()
        assert data["error_code"] == "UNAUTHORIZED"


@pytest.mark.asyncio
async def test_empty_audio_payload_returns_400():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("audio.wav", b"", "audio/wav")}
        headers = {"X-API-Key": settings.api_key}
        response = await ac.post("/v1/analyze", files=files, headers=headers)
        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == "EMPTY_AUDIO_PAYLOAD"


@pytest.mark.asyncio
async def test_audio_payload_too_large_returns_413(monkeypatch):
    monkeypatch.setattr(settings, "max_audio_bytes", 100)  # Set tiny limit
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("audio.wav", b"x" * 200, "audio/wav")}
        headers = {"X-API-Key": settings.api_key}
        response = await ac.post("/v1/analyze", files=files, headers=headers)
        assert response.status_code == 413
        data = response.json()
        assert data["error_code"] == "AUDIO_PAYLOAD_TOO_LARGE"
        assert data["extra"]["max_bytes"] == 100


@pytest.mark.asyncio
async def test_rate_limiting_enforcement(monkeypatch):
    # Set limit to 2 requests per 60 seconds
    monkeypatch.setattr(rate_limiter, "max_requests", 2)
    monkeypatch.setattr(rate_limiter, "window_seconds", 60)

    # Mock analyze_audio to avoid external network waiting
    async def fast_mock_analyze(audio, filename, content_type, request_id):
        return AnalysisResponse(
            request_id=request_id,
            final_outcome=FinalOutcome.INCONCLUSIVE,
            reason_code="MOCK_TEST",
            pella_status="skipped",
            rias_transcription_status="skipped",
            rias_risk_status="skipped",
        )

    monkeypatch.setattr("app.api.routes.analyze_audio", fast_mock_analyze)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("audio.wav", b"valid audio bytes", "audio/wav")}
        headers = {"X-API-Key": settings.api_key}

        # Request 1: allowed
        res1 = await ac.post("/v1/analyze", files=files, headers=headers)
        assert res1.status_code == 200

        # Request 2: allowed
        res2 = await ac.post("/v1/analyze", files=files, headers=headers)
        assert res2.status_code == 200

        # Request 3: rate limited!
        res3 = await ac.post("/v1/analyze", files=files, headers=headers)
        assert res3.status_code == 429
        data = res3.json()
        assert data["error_code"] == "RATE_LIMIT_EXCEEDED"
        assert "Retry-After" in res3.headers


@pytest.mark.asyncio
async def test_custom_request_id_preserved(monkeypatch):
    custom_id = "custom-trace-uuid-9999"

    async def fast_mock_analyze(audio, filename, content_type, request_id):
        return AnalysisResponse(
            request_id=request_id,
            final_outcome=FinalOutcome.INCONCLUSIVE,
            reason_code="MOCK_TEST",
            pella_status="skipped",
            rias_transcription_status="skipped",
            rias_risk_status="skipped",
        )

    monkeypatch.setattr("app.api.routes.analyze_audio", fast_mock_analyze)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("audio.wav", b"audio payload bytes", "audio/wav")}
        headers = {"X-API-Key": settings.api_key, "X-Request-ID": custom_id}
        response = await ac.post("/v1/analyze", files=files, headers=headers)
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_id
        assert response.json()["request_id"] == custom_id
