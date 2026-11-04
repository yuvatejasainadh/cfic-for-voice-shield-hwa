"""Service contract tests for PellaV2 detector HTTP client."""

import pytest
import httpx
from services.pella.client import PellaClient


class TestPellaClientContract:
    @pytest.mark.asyncio
    async def test_unconfigured_base_url_returns_not_configured(self):
        client = PellaClient(base_url="")
        res = await client.predict(b"dummy audio")
        assert res["status"] == "not_configured"
        assert res["prediction"] is None
        assert res["latency_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_successful_predict_response_parsing(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            return httpx.Response(
                status_code=200,
                json={
                    "status": "success",
                    "prediction": "GENUINE",
                    "score": 0.08,
                    "model_version": "pellav2-quantized-int8",
                    "latency_ms": 28.0,
                },
                request=httpx.Request("POST", "http://pella-service:8101/predict"),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        client = PellaClient(base_url="http://pella-service:8101")
        res = await client.predict(b"sample audio")
        assert res["status"] == "success"
        assert res["prediction"] == "GENUINE"
        assert res["score"] == 0.08
        assert res["model_version"] == "pellav2-quantized-int8"

    @pytest.mark.asyncio
    async def test_upstream_timeout_handled(self, monkeypatch):
        async def mock_timeout(*args, **kwargs):
            raise httpx.TimeoutException("Connection timed out")

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_timeout)

        client = PellaClient(base_url="http://pella-service:8101")
        res = await client.predict(b"sample audio")
        assert res["status"] == "timeout"
        assert "timed out" in res["error"]
