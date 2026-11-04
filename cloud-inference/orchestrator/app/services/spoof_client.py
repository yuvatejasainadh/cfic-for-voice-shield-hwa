"""Client communicating with VMC-South and AASIST-L spoof detector services."""

import time
from typing import Any
import httpx
from app.core.config import settings


class SpoofClient:
    """Async client invoking acoustic spoof detectors."""

    def __init__(self, timeout: float | None = None):
        self.timeout = timeout or settings.model_timeout_seconds

    async def predict(
        self,
        endpoint_url: str,
        audio: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        start = time.perf_counter()
        endpoint = f"{endpoint_url.rstrip('/')}/predict"
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
                    "prediction": None,
                    "score": None,
                    "error": f"Spoof detector returned HTTP {res.status_code}",
                    "latency_ms": elapsed_ms,
                }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "status": "error",
                "prediction": None,
                "score": None,
                "error": str(exc),
                "latency_ms": elapsed_ms,
            }


spoof_client = SpoofClient()
