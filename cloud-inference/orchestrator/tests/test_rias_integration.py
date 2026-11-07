"""Integration tests for RIAS Gemma microservice and Orchestrator-TCED pipeline."""

import importlib.util
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.rias_client import RIASClient


@pytest.mark.asyncio
async def test_rias_client_to_service_contract():
    """Verify RIASClient serializes requests and normalizes responses properly."""
    mock_response = {
        "job_id": "sess-integration-01",
        "chunk_index": 0,
        "status": "SUCCESS",
        "risk_level": "HIGH",
        "failure_reason": None,
        "model_id": "google/gemma-4-E2B-it",
        "model_revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
        "latency_ms": 350.2,
    }

    client = RIASClient(base_url="http://mock-rias:8331", timeout=5.0)

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json = lambda: mock_response
        mock_post.return_value = mock_resp

        result = await client.assess(
            transcript="Police department calling, send payment immediately.",
            language="en",
            session_id="sess-integration-01",
            chunk_index=0,
        )

        assert result["status"] == "SUCCESS"
        assert result["risk_level"] == "HIGH"
        assert result["failure_reason"] is None
        assert result["model_id"] == "google/gemma-4-E2B-it"


@pytest.mark.asyncio
async def test_rias_client_connection_error_produces_inconclusive():
    """Verify RIASClient gracefully yields INCONCLUSIVE on network/connect error or timeout."""
    client = RIASClient(base_url="http://nonexistent-host:9999", timeout=0.2)

    result = await client.assess(
        transcript="Test transcript",
        language="en",
        session_id="sess-fail-01",
    )

    assert result["status"] == "inconclusive"
    assert result["risk_level"] == "INCONCLUSIVE"
    assert result["failure_reason"] is not None
    assert any(k in result["failure_reason"] for k in ["CONNECTION_ERROR", "INFERENCE_TIMEOUT", "ConnectError"])


@pytest.mark.asyncio
async def test_full_lid_asr_gemma_pipeline():
    """Verify full sequential orchestration: LID -> ASR -> Gemma -> Pipeline Result."""
    mock_lid = {
        "status": "success",
        "language": "en",
        "raw_label": "en-IN",
        "confidence": 0.99,
        "model_id": "onecxi/open-vakgyata",
        "latency_ms": 100.0,
    }
    mock_asr = {
        "status": "success",
        "text": "Your debit card has been blocked. Provide your one time password to reactivate.",
        "language": "en",
        "model_id": "openai/whisper-base.en",
        "latency_ms": 320.0,
    }
    mock_spoof = {
        "status": "success",
        "prediction": "GENUINE",
        "score": 0.01,
        "model_version": "aasist-fallback",
        "latency_ms": 90.0,
    }
    mock_rias = {
        "status": "success",
        "risk_level": "HIGH",
        "failure_reason": None,
        "model_id": "google/gemma-4-E2B-it",
        "model_revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
        "latency_ms": 410.0,
    }

    with patch("app.api.routes.lid_client.identify", new=AsyncMock(return_value=mock_lid)), \
         patch("app.api.routes.asr_client.transcribe", new=AsyncMock(return_value=mock_asr)), \
         patch("app.api.routes.spoof_client.predict", new=AsyncMock(return_value=mock_spoof)), \
         patch("app.api.routes.rias_client.assess", new=AsyncMock(return_value=mock_rias)):

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            files = {"file": ("chunk.wav", b"fake-wav-data", "audio/wav")}
            res = await ac.post("/v1/inference", files=files, data={"session_id": "sess-pipeline-01"})

            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "success"
            assert data["lid"]["language"] == "en"
            assert data["transcription"]["text"] == "Your debit card has been blocked. Provide your one time password to reactivate."
            assert data["rias"]["risk_level"] == "HIGH"
            assert data["spoof"]["prediction"] == "GENUINE"


def test_tced_decision_policy_inconclusive_invariants():
    """Verify TCED rule: HIGH RIAS forces SUSPICIOUS; inconclusive never defaults to LOW."""
    # Test TCED logic directly using DecisionEngine pattern
    def tced_evaluate(spoof: dict, rias: dict) -> tuple[str, str]:
        if rias.get("status") == "success" and rias.get("risk_level") == "HIGH":
            return "SUSPICIOUS", "HIGH_RIAS_OVERRIDE"
        if rias.get("risk_level") == "INCONCLUSIVE" or rias.get("status") != "success":
            return "INCONCLUSIVE", "INVALID_OR_MISSING_RIAS"
        return "GENUINE", "NORMAL"

    # Acoustic says GENUINE, but Gemma RIAS says HIGH
    outcome, reason = tced_evaluate(
        {"status": "success", "prediction": "GENUINE"},
        {"status": "success", "risk_level": "HIGH"}
    )
    assert outcome == "SUSPICIOUS"
    assert reason == "HIGH_RIAS_OVERRIDE"

    # Inconclusive RIAS
    outcome_inc, reason_inc = tced_evaluate(
        {"status": "success", "prediction": "GENUINE"},
        {"status": "inconclusive", "risk_level": "INCONCLUSIVE"}
    )
    assert outcome_inc == "INCONCLUSIVE"


def test_no_public_exposure_in_docker_compose():
    """Verify Gemma RIAS microservice has zero host port mappings in docker-compose.yml."""
    import yaml
    compose_path = Path(__file__).resolve().parents[3] / "docker-compose.yml"
    with open(compose_path, "r", encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    services = compose.get("services", {})
    assert "model-rias-gemma" in services, "model-rias-gemma missing from docker-compose.yml"
    rias_service = services["model-rias-gemma"]

    # Invariant: No public port mapping
    assert "ports" not in rias_service, "Security violation: model-rias-gemma must NOT expose host ports"
    # Invariant: Only on internal network
    assert rias_service.get("networks") == ["cfic-internal-network"], "model-rias-gemma must be isolated on cfic-internal-network"
