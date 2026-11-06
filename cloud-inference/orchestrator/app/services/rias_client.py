"""Client communicating with the Gemma 4 E2B IT RIAS LLM microservice."""

import logging
import time
from typing import Any, Dict, Optional
import httpx
from app.core.config import settings

logger = logging.getLogger("rias_client")


class RIASClient:
    """Async client invoking internal Gemma 4 E2B IT microservice for semantic risk assessment."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = (base_url or settings.rias_service_url).rstrip("/")
        self.timeout = timeout or settings.rias_timeout_seconds

    async def assess(
        self,
        transcript: str,
        language: str = "en",
        session_id: Optional[str] = None,
        chunk_index: int = 0,
    ) -> Dict[str, Any]:
        """Send transcript to Gemma RIAS microservice."""
        clean_text = (transcript or "").strip()
        if not clean_text:
            return {
                "status": "skipped",
                "risk_level": "INCONCLUSIVE",
                "failure_reason": "EMPTY_TRANSCRIPT",
                "model_id": "google/gemma-4-E2B-it",
                "model_revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
                "latency_ms": 0.0,
            }

        start = time.perf_counter()
        endpoint = f"{self.base_url}/v1/assess"
        payload = {
            "job_id": session_id or "unknown",
            "chunk_index": chunk_index,
            "language": language,
            "transcript": clean_text,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(endpoint, json=payload)
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                if res.status_code == 200:
                    data = res.json()
                    data["latency_ms"] = elapsed_ms
                    return data

                return {
                    "status": "inconclusive",
                    "risk_level": "INCONCLUSIVE",
                    "failure_reason": f"HTTP_{res.status_code}: {res.text[:80]}",
                    "model_id": "google/gemma-4-E2B-it",
                    "model_revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
                    "latency_ms": elapsed_ms,
                }
        except httpx.TimeoutException:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(f"RIAS request timed out after {self.timeout}s for session {session_id}")
            return {
                "status": "inconclusive",
                "risk_level": "INCONCLUSIVE",
                "failure_reason": "INFERENCE_TIMEOUT",
                "model_id": "google/gemma-4-E2B-it",
                "model_revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
                "latency_ms": elapsed_ms,
            }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(f"RIAS client connection error: {type(exc).__name__}: {str(exc)}")
            return {
                "status": "inconclusive",
                "risk_level": "INCONCLUSIVE",
                "failure_reason": f"CONNECTION_ERROR: {type(exc).__name__}",
                "model_id": "google/gemma-4-E2B-it",
                "model_revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
                "latency_ms": elapsed_ms,
            }


rias_client = RIASClient()
