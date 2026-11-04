"""Client communicating with the language-specific ASR microservices."""

import logging
import time
from typing import Any, Dict
import httpx
from app.core.config import settings

logger = logging.getLogger("asr_client")


class ASRClient:
    """Async client invoking language-specific ASR instances (English Whisper / IndicConformer)."""

    def __init__(self, timeout: float | None = None):
        self.timeout = timeout or settings.asr_timeout_seconds

    async def transcribe(
        self,
        endpoint_url: str,
        audio: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> Dict[str, Any]:
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
                    "error": f"ASR returned HTTP {res.status_code}: {res.text}",
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
