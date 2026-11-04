"""Tests for Cloud Inference Orchestrator."""

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.routing.language_router import language_router
from app.routing.spoof_selector import spoof_selector


def test_language_router_selection():
    # English
    url, lang = language_router.select_asr_endpoint("en")
    assert lang == "en"

    # Telugu
    url, lang = language_router.select_asr_endpoint("te")
    assert lang == "te"

    # Hindi
    url, lang = language_router.select_asr_endpoint("hi")
    assert lang == "hi"

    # South Indian cluster fallback
    url, lang = language_router.select_asr_endpoint("ta")
    assert lang == "ta"


def test_spoof_selector():
    # South Indian languages select VMC-South
    for lang in ["te", "ta", "kn", "ml"]:
        url, model = spoof_selector.select_spoof_endpoint(lang)
        assert model == "vmc-south"

    # English and general fallback select AASIST-L fallback
    for lang in ["en", "hi", "mr", "unknown"]:
        url, model = spoof_selector.select_spoof_endpoint(lang)
        assert model == "aasist-fallback"


@pytest.mark.asyncio
async def test_orchestrator_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["service"] == "cloud-inference-orchestrator"
        assert "lid" in data["backends"]
        assert "asr_en" in data["backends"]
        assert "vmc_south" in data["backends"]
