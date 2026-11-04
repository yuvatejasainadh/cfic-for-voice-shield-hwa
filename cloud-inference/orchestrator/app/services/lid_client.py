"""Client communicating with the Language Identification (LID) microservice."""

import time
from typing import Any
import httpx
from app.core.config import settings


class LIDClient:
    """Async client invoking LID service."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or settings.lid_service_url).rstrip("/")
        self.timeout = timeout or settings.model_timeout_seconds

    async def identify(
        self, audio: bytes, filename: str = "audio.wav", content_type: str = "audio/wav"
    ) -> dict[str, Any]:
        start = time.perf_counter()
        endpoint = f"{self.base_url}/identify"
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
                    "language": "en",
                    "confidence": 0.0,
                    "error": f"LID returned HTTP {res.status_code}",
                    "latency_ms": elapsed_ms,
                }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "status": "error",
                "language": "en",  # fallback default language
                "confidence": 0.0,
                "error": str(exc),
                "latency_ms": elapsed_ms,
            }


lid_client = LIDClient()
