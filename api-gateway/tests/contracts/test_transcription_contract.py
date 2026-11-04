"""Service contract tests for Speech Transcription HTTP client."""

import pytest
import httpx
from services.transcription.client import TranscriptionClient


class TestTranscriptionClientContract:
    @pytest.mark.asyncio
    async def test_unconfigured_base_url_returns_not_configured(self):
        client = TranscriptionClient(base_url="")
        res = await client.transcribe(b"dummy audio")
        assert res["status"] == "not_configured"
        assert res["text"] is None
        assert res["latency_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_successful_transcribe_response_parsing(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            return httpx.Response(
                status_code=200,
                json={
                    "status": "success",
                    "text": "Hello this is a customer support call.",
                    "language": "en",
                    "model_version": "whisper-indic-v1",
                    "latency_ms": 52.0,
                },
                request=httpx.Request("POST", "http://transcription-service:8102/transcribe"),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        client = TranscriptionClient(base_url="http://transcription-service:8102")
        res = await client.transcribe(b"sample audio")
        assert res["status"] == "success"
        assert res["text"] == "Hello this is a customer support call."
        assert res["language"] == "en"
        assert res["model_version"] == "whisper-indic-v1"

    @pytest.mark.asyncio
    async def test_upstream_timeout_handled(self, monkeypatch):
        async def mock_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Connection timed out")

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_timeout)

        client = TranscriptionClient(base_url="http://transcription-service:8102")
        res = await client.transcribe(b"sample audio")
        assert res["status"] == "timeout"
        assert "timed out" in res["error"]
