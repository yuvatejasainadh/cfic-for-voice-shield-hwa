"""HTTP client for the downstream PellaV2 audio deepfake detector service."""

import time
from typing import Any
import httpx
from app.core.config import settings


class PellaClient:
    """Async client communicating with the PellaV2 detector service."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        raw_url = base_url if base_url is not None else (settings.pella_base_url or "")
        self.base_url = raw_url.rstrip("/") if raw_url else ""
        self.timeout = timeout if timeout is not None else settings.request_timeout_seconds

    async def predict(
        self, audio: bytes, filename: str = "audio.wav", content_type: str = "audio/wav"
    ) -> dict[str, Any]:
        """Send audio to PellaV2 detector for inference.

        Returns:
            dict containing:
                - status: 'success' | 'error' | 'timeout' | 'not_configured'
                - prediction: str | None (e.g., 'GENUINE', 'FAKE', or native artifact label)
                - score: float | None
                - model_version: str | None
                - error: str | None
                - latency_ms: float
        """
        start = time.perf_counter()

        if not self.base_url:
            return {
                "status": "not_configured",
                "prediction": None,
                "score": None,
                "model_version": None,
                "error": "PellaV2 service URL is not configured",
                "latency_ms": 0.0,
            }

        endpoint = f"{self.base_url}/predict"
        files = {"file": (filename, audio, content_type)}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint, files=files)
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

                if response.status_code != 200:
                    return {
                        "status": "error",
                        "prediction": None,
                        "score": None,
                        "model_version": None,
                        "error": f"Upstream error HTTP {response.status_code}",
                        "latency_ms": elapsed_ms,
                    }

                data = response.json()
                return {
                    "status": "success",
                    "prediction": data.get("prediction"),
                    "score": data.get("score"),
                    "model_version": data.get("model_version", "pellav2-unknown"),
                    "error": None,
                    "latency_ms": elapsed_ms,
                }
        except httpx.TimeoutException:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "status": "timeout",
                "prediction": None,
                "score": None,
                "model_version": None,
                "error": "PellaV2 detector request timed out",
                "latency_ms": elapsed_ms,
            }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "status": "error",
                "prediction": None,
                "score": None,
                "model_version": None,
                "error": f"PellaV2 request failed: {type(exc).__name__}",
                "latency_ms": elapsed_ms,
            }
