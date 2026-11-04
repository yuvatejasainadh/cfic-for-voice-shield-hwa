"""Client communicating with the Language Identification (LID) microservice."""

import asyncio
import logging
import time
from typing import Any, Dict
import httpx
from app.core.config import settings

logger = logging.getLogger("lid_client")


class LIDClient:
    """Async client invoking Open-Vakgyata LID microservice with retry support."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or settings.lid_service_url).rstrip("/")
        self.timeout = timeout or settings.lid_timeout_seconds

    async def identify(
        self,
        audio: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        max_retries: int | None = None,
    ) -> Dict[str, Any]:
        """Call Open-Vakgyata LID service with retry capability."""
        endpoint = f"{self.base_url}/identify"
        retries = max_retries if max_retries is not None else settings.lid_max_retries
        attempts = 1 + retries

        last_error = None
        start = time.perf_counter()

        for attempt in range(attempts):
            try:
                files = {"file": (filename, audio, content_type)}
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    res = await client.post(endpoint, files=files)
                    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                    if res.status_code == 200:
                        data = res.json()
                        data["retries_used"] = attempt
                        data["latency_ms"] = elapsed_ms
                        return data

                    last_error = f"LID returned HTTP {res.status_code}: {res.text}"
                    logger.warning(f"LID attempt {attempt + 1}/{attempts} failed: {last_error}")
            except Exception as exc:
                last_error = str(exc)
                logger.warning(f"LID attempt {attempt + 1}/{attempts} encountered exception: {exc}")

            if attempt < attempts - 1:
                await asyncio.sleep(0.1 * (attempt + 1))

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "status": "error",
            "language": "unknown",
            "raw_label": None,
            "confidence": 0.0,
            "probabilities": {},
            "top_k": [],
            "model_id": "onecxi/open-vakgyata",
            "model_revision": None,
            "retries_used": retries,
            "error": last_error,
            "latency_ms": elapsed_ms,
        }


lid_client = LIDClient()
