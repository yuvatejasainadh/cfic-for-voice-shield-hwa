"""HTTP client for the downstream transcription service."""

import time
from typing import Any
import httpx
from app.core.config import settings


class TranscriptionClient:
    """Async client communicating with the speech transcription service."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        raw_url = base_url if base_url is not None else (settings.transcription_base_url or "")
        self.base_url = raw_url.rstrip("/") if raw_url else ""
        self.timeout = timeout if timeout is not None else settings.request_timeout_seconds

    async def transcribe(
        self, audio: bytes, filename: str = "audio.wav", content_type: str = "audio/wav"
    ) -> dict[str, Any]:
        """Send audio to transcription service for speech-to-text.

        Returns:
            dict containing:
                - status: 'success' | 'error' | 'timeout' | 'not_configured'
                - text: str | None
                - language: str | None
                - model_version: str | None
                - error: str | None
                - latency_ms: float
        """
        start = time.perf_counter()

        if not self.base_url:
            return {
                "status": "not_configured",
                "text": None,
                "language": None,
                "model_version": None,
                "error": "Transcription service URL is not configured",
                "latency_ms": 0.0,
            }

        endpoint = f"{self.base_url}/transcribe"
        files = {"file": (filename, audio, content_type)}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, files=files)
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

                if response.status_code != 200:
                    return {
                        "status": "error",
                        "text": None,
                        "language": None,
                        "model_version": None,
                        "error": f"Upstream error HTTP {response.status_code}",
                        "latency_ms": elapsed_ms,
                    }

                data = response.json()
                return {
                    "status": "success",
                    "text": data.get("text"),
                    "language": data.get("language"),
                    "model_version": data.get("model_version", "transcription-unknown"),
                    "error": None,
                    "latency_ms": elapsed_ms,
                }
        except httpx.TimeoutException:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "status": "timeout",
                "text": None,
                "language": None,
                "model_version": None,
                "error": "Transcription request timed out",
                "latency_ms": elapsed_ms,
            }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "status": "error",
                "text": None,
                "language": None,
                "model_version": None,
                "error": f"Transcription request failed: {type(exc).__name__}",
                "latency_ms": elapsed_ms,
            }
