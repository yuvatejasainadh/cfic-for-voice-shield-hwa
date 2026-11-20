"""Integration tests for PellaV2 K12 INT8 acoustic spoof detector in Cloud Inference Orchestrator."""

import pytest
from httpx import Response
from app.core.config import settings
from app.routing.spoof_selector import spoof_selector
from app.services.spoof_client import spoof_client


class TestPellaOrchestratorIntegration:
    def test_spoof_selector_prefers_pellav2_when_configured(self, monkeypatch):
        monkeypatch.setattr(settings, "pellav2_service_url", "http://model-pellav2:8321")
        
        endpoint, model_name = spoof_selector.select_spoof_endpoint("te")
        assert endpoint == "http://model-pellav2:8321"
        assert model_name == "pellav2-k12-int8"

        endpoint_en, model_name_en = spoof_selector.select_spoof_endpoint("en")
        assert endpoint_en == "http://model-pellav2:8321"
        assert model_name_en == "pellav2-k12-int8"

    def test_spoof_selector_preserves_regional_fallback_when_pellav2_unconfigured(self, monkeypatch):
        monkeypatch.setattr(settings, "pellav2_service_url", "")
        
        endpoint_te, model_te = spoof_selector.select_spoof_endpoint("te")
        assert endpoint_te == settings.vmc_south_service_url
        assert model_te == "vmc-south"

        endpoint_en, model_en = spoof_selector.select_spoof_endpoint("en")
        assert endpoint_en == settings.aasist_fallback_service_url
        assert model_en == "aasist-fallback"

    @pytest.mark.asyncio
    async def test_pella_client_predict_success(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            return Response(
                status_code=200,
                json={
                    "status": "success",
                    "prediction": "FAKE",
                    "score": 0.8837,
                    "threshold": 0.50,
                    "model_version": "pellav2-k12-int8",
                    "latency_ms": 35.2,
                },
            )

        import httpx
        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        res = await spoof_client.predict("http://model-pellav2:8321", b"fake-audio")
        assert res["status"] == "success"
        assert res["prediction"] == "FAKE"
        assert res["score"] == 0.8837
        assert res["model_version"] == "pellav2-k12-int8"

    @pytest.mark.asyncio
    async def test_pella_client_failure_returns_error(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            return Response(status_code=503, text="Service Unavailable")

        import httpx
        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        res = await spoof_client.predict("http://model-pellav2:8321", b"audio")
        assert res["status"] == "error"
        assert res["prediction"] is None
        assert "503" in res["error"]
