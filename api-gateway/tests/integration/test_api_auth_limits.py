"""Integration tests for authentication, payload limits, rate limiting, and concurrency control."""

import asyncio
import io
import time
import wave
import pytest
from httpx import ASGITransport, AsyncClient
from app.api.routes import rate_limiter
from app.core.concurrency import concurrency_manager
from app.core.config import settings
from app.main import app
from app.schemas import AnalysisResponse, FinalOutcome


def make_test_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate minimal valid PCM 16-bit mono WAV audio in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(sample_rate * duration_s))
    return buf.getvalue()


@pytest.fixture(autouse=True)
def reset_limiter_and_concurrency():
    rate_limiter.reset()
    concurrency_manager.reset()


@pytest.mark.asyncio
async def test_missing_api_key_returns_401():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("audio.wav", make_test_wav(1.0), "audio/wav")}
        response = await ac.post("/v1/analyze", files=files)
        assert response.status_code == 401
        data = response.json()
        assert data["error_code"] == "UNAUTHORIZED"
        assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_invalid_api_key_returns_401():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("audio.wav", make_test_wav(1.0), "audio/wav")}
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
async def test_unsupported_audio_mime_returns_415():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("document.txt", b"plain text content", "text/plain")}
        headers = {"X-API-Key": settings.api_key}
        response = await ac.post("/v1/analyze", files=files, headers=headers)
        assert response.status_code == 415
        data = response.json()
        assert data["error_code"] == "UNSUPPORTED_AUDIO_FORMAT"


@pytest.mark.asyncio
async def test_unsupported_audio_extension_returns_415():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("audio.pdf", make_test_wav(1.0), "audio/wav")}
        headers = {"X-API-Key": settings.api_key}
        response = await ac.post("/v1/analyze", files=files, headers=headers)
        assert response.status_code == 415
        data = response.json()
        assert data["error_code"] == "UNSUPPORTED_AUDIO_FORMAT"


@pytest.mark.asyncio
async def test_corrupted_audio_bitstream_returns_400():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("corrupted.wav", b"not-a-real-audio-stream", "audio/wav")}
        headers = {"X-API-Key": settings.api_key}
        response = await ac.post("/v1/analyze", files=files, headers=headers)
        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == "CORRUPTED_AUDIO_PAYLOAD"


@pytest.mark.asyncio
async def test_audio_duration_exceeding_max_returns_400(monkeypatch):
    monkeypatch.setattr(settings, "max_audio_duration_seconds", 2.0)
    # Generate 3 seconds of WAV
    long_wav = make_test_wav(duration_s=3.0)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("too_long.wav", long_wav, "audio/wav")}
        headers = {"X-API-Key": settings.api_key}
        response = await ac.post("/v1/analyze", files=files, headers=headers)
        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == "AUDIO_DURATION_TOO_LONG"
        assert data["extra"]["max_duration_seconds"] == 2.0
        assert data["extra"]["actual_duration_seconds"] >= 3.0


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
        files = {"file": ("audio.wav", make_test_wav(1.0), "audio/wav")}
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
        files = {"file": ("audio.wav", make_test_wav(1.0), "audio/wav")}
        headers = {"X-API-Key": settings.api_key, "X-Request-ID": custom_id}
        response = await ac.post("/v1/analyze", files=files, headers=headers)
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_id
        assert response.json()["request_id"] == custom_id


@pytest.mark.asyncio
async def test_concurrency_queue_saturation_returns_503(monkeypatch):
    # Set concurrency to 1 and queue to 0 so 2nd concurrent request fails immediately
    concurrency_manager.configure(max_concurrency=1, max_queue=0)

    slow_barrier = asyncio.Event()

    async def blocking_analyze(audio, filename, content_type, request_id):
        await slow_barrier.wait()
        return AnalysisResponse(
            request_id=request_id,
            final_outcome=FinalOutcome.INCONCLUSIVE,
            reason_code="BLOCKING_MOCK",
            pella_status="skipped",
            rias_transcription_status="skipped",
            rias_risk_status="skipped",
        )

    monkeypatch.setattr("app.api.routes.analyze_audio", blocking_analyze)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("audio.wav", make_test_wav(1.0), "audio/wav")}
        headers = {"X-API-Key": settings.api_key}

        # Launch request 1 (holds the 1 execution slot)
        task1 = asyncio.create_task(ac.post("/v1/analyze", files=files, headers=headers))
        await asyncio.sleep(0.05)  # Ensure task1 acquires slot

        # Request 2 arrives: queue capacity is 0 -> 503
        res2 = await ac.post("/v1/analyze", files=files, headers=headers)
        assert res2.status_code == 503
        data2 = res2.json()
        assert data2["error_code"] == "SERVICE_OVERLOADED"
        assert "Retry-After" in res2.headers

        # Release task1
        slow_barrier.set()
        res1 = await task1
        assert res1.status_code == 200


@pytest.mark.asyncio
async def test_concurrency_queue_timeout_returns_504(monkeypatch):
    # Set queue timeout to 0.1s
    concurrency_manager.configure(max_concurrency=1, max_queue=5, queue_timeout=0.1)

    slow_barrier = asyncio.Event()

    async def blocking_analyze(audio, filename, content_type, request_id):
        await slow_barrier.wait()
        return AnalysisResponse(
            request_id=request_id,
            final_outcome=FinalOutcome.INCONCLUSIVE,
            reason_code="TIMEOUT_MOCK",
            pella_status="skipped",
            rias_transcription_status="skipped",
            rias_risk_status="skipped",
        )

    monkeypatch.setattr("app.api.routes.analyze_audio", blocking_analyze)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("audio.wav", make_test_wav(1.0), "audio/wav")}
        headers = {"X-API-Key": settings.api_key}

        # Launch request 1 (holds slot)
        task1 = asyncio.create_task(ac.post("/v1/analyze", files=files, headers=headers))
        await asyncio.sleep(0.05)

        # Request 2 enters queue, waits >0.1s -> 504
        res2 = await ac.post("/v1/analyze", files=files, headers=headers)
        assert res2.status_code == 504
        data2 = res2.json()
        assert data2["error_code"] == "QUEUE_TIMEOUT"

        # Release task1
        slow_barrier.set()
        res1 = await task1
        assert res1.status_code == 200


@pytest.mark.asyncio
async def test_rate_limiter_stale_key_cleanup():
    limiter = rate_limiter
    # Insert keys directly with past timestamps
    past_time = time.time() - 100.0
    limiter._requests["stale-client-1"].append(past_time)
    limiter._requests["stale-client-2"].append(past_time)
    limiter._requests["active-client"].append(time.time())

    assert "stale-client-1" in limiter._requests
    limiter._cleanup_expired()
    assert "stale-client-1" not in limiter._requests
    assert "stale-client-2" not in limiter._requests
    assert "active-client" in limiter._requests
