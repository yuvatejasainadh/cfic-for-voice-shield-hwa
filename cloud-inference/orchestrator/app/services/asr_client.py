"""Client communicating with the IndicConformer ASR microservices."""

import time
from typing import Any
import httpx
from app.core.config import settings


class ASRClient:
    """Async client invoking language-specific IndicConformer instances."""

    def __init__(self, timeout: float | None = None):
        self.timeout = timeout or settings.model_timeout_seconds

    async def transcribe(
        self,
        endpoint_url: str,
        audio: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        start = time.perf_counter()
        endpoint = f"{endpoint_url.rstrip('/')}/transcribe"
        files = {"file": (filename, audio, content_type)}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(endpoint, files=files)
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                if res.status_code == 200:
                    data = res.json()
                    data["latency_ms"] = elapsed_ms
                    return data
                return {
                    "status": "error",
                    "text": None,
                    "language": "en",
                    "error": f"ASR returned HTTP {res.status_code}",
                    "latency_ms": elapsed_ms,
                }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "status": "error",
                "text": None,
                "language": "en",
                "error": str(exc),
                "latency_ms": elapsed_ms,
            }


asr_client = ASRClient()
