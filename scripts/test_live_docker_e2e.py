"""Comprehensive Live Docker E2E & Risk Assessment Policy Verification."""

import asyncio
import io
import os
import sys
import httpx
import pytest

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import api-gateway decision engine and schemas
api_gw_path = os.path.join(repo_root, "api-gateway")
if api_gw_path not in sys.path:
    sys.path.insert(0, api_gw_path)

from app.schemas.analysis import FinalOutcome, RiskLevel
from app.orchestration.decision_engine import DecisionEngine


DOCKER_PELLAV2_URL = "http://localhost:8321"
TEST_AUDIO_PATH = os.path.join(repo_root, "cfic-asr-benchmark", "data", "audio", "telugu", "telugu_00000.wav")


def test_docker_live_health():
    """Verify live Docker container health and readiness."""
    with httpx.Client(base_url=DOCKER_PELLAV2_URL, timeout=10.0) as client:
        r_health = client.get("/health")
        assert r_health.status_code == 200
        health_data = r_health.json()
        assert health_data["status"] == "ok"
        assert health_data["service"] == "model-pellav2"
        assert health_data["model_version"] == "pellav2-k12-int8"
        assert health_data["is_mock"] is False
        assert health_data["checksum_verified"] is True

        r_ready = client.get("/ready")
        assert r_ready.status_code == 200
        ready_data = r_ready.json()
        assert ready_data["ready"] is True
        assert ready_data["model_loaded"] is True


def test_docker_live_real_audio_inference():
    """Verify live Docker INT8 inference on real audio."""
    assert os.path.exists(TEST_AUDIO_PATH), f"Audio not found: {TEST_AUDIO_PATH}"
    with open(TEST_AUDIO_PATH, "rb") as f:
        audio_bytes = f.read()

    with httpx.Client(base_url=DOCKER_PELLAV2_URL, timeout=15.0) as client:
        files = {"file": ("telugu_00000.wav", audio_bytes, "audio/wav")}
        r = client.post("/predict", files=files)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["prediction"] == "FAKE"
        assert data["score"] > 0.50
        assert data["threshold"] == 0.50
        assert data["model_version"] == "pellav2-k12-int8"
        assert data["latency_ms"] > 0


def test_docker_live_empty_audio_rejected():
    """Docker container must reject empty audio with 400 Bad Request."""
    with httpx.Client(base_url=DOCKER_PELLAV2_URL, timeout=10.0) as client:
        files = {"file": ("empty.wav", b"", "audio/wav")}
        r = client.post("/predict", files=files)
        assert r.status_code == 400
        assert "Empty audio payload" in r.json()["detail"]


def test_docker_live_corrupted_audio_rejected():
    """Docker container must reject corrupt audio bytes with 400 Bad Request."""
    with httpx.Client(base_url=DOCKER_PELLAV2_URL, timeout=10.0) as client:
        files = {"file": ("corrupted.wav", b"NOT_A_WAV_HEADER_DATA_12345", "audio/wav")}
        r = client.post("/predict", files=files)
        assert r.status_code == 400


def test_docker_live_oversized_audio_rejected():
    """Docker container must reject payloads exceeding 25MB with 413."""
    oversized = b"0" * (26 * 1024 * 1024)
    with httpx.Client(base_url=DOCKER_PELLAV2_URL, timeout=10.0) as client:
        files = {"file": ("huge.wav", oversized, "audio/wav")}
        r = client.post("/predict", files=files)
        assert r.status_code == 413


@pytest.mark.asyncio
async def test_orchestrator_spoof_client_contract_against_live_docker():
    """Verify orchestrator spoof contract against live Docker container."""
    with open(TEST_AUDIO_PATH, "rb") as f:
        audio_bytes = f.read()

    async with httpx.AsyncClient(timeout=15.0) as client:
        files = {"file": ("telugu_00000.wav", audio_bytes, "audio/wav")}
        r = await client.post(f"{DOCKER_PELLAV2_URL}/predict", files=files)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["prediction"] == "FAKE"
        assert data["score"] > 0.50
        assert data["model_version"] == "pellav2-k12-int8"


# ==============================================================================
# Phase 6: Risk Assessment & TCED Policy Verification
# ==============================================================================

def test_tced_row1_fake_low():
    """Acoustic FAKE + RIAS LOW -> FAKE (ACOUSTIC_SPOOF_DETECTED)."""
    engine = DecisionEngine(calibrated=True)
    outcome, reason = engine.decide(
        {"status": "success", "prediction": "FAKE", "score": 0.9156},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome == FinalOutcome.FAKE
    assert reason == "ACOUSTIC_SPOOF_DETECTED"


def test_tced_row2_genuine_low():
    """Acoustic GENUINE + RIAS LOW -> GENUINE (ACOUSTIC_GENUINE_CONFIRMED)."""
    engine = DecisionEngine(calibrated=True)
    outcome, reason = engine.decide(
        {"status": "success", "prediction": "GENUINE", "score": 0.3304},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome == FinalOutcome.GENUINE
    assert reason == "ACOUSTIC_GENUINE_CONFIRMED"


def test_tced_row3_genuine_high():
    """Acoustic GENUINE + RIAS HIGH -> SUSPICIOUS (HIGH_RIAS_OVERRIDE)."""
    engine = DecisionEngine(calibrated=True)
    outcome, reason = engine.decide(
        {"status": "success", "prediction": "GENUINE", "score": 0.3304},
        {"status": "success", "risk_level": "HIGH"},
    )
    assert outcome == FinalOutcome.SUSPICIOUS
    assert reason == "HIGH_RIAS_OVERRIDE"


def test_tced_row4_detector_unavailable_high():
    """Detector unavailable + RIAS HIGH -> SUSPICIOUS (HIGH_RIAS_OVERRIDE)."""
    engine = DecisionEngine(calibrated=True)
    outcome, reason = engine.decide(
        {"status": "error", "prediction": None},
        {"status": "success", "risk_level": "HIGH"},
    )
    assert outcome == FinalOutcome.SUSPICIOUS
    assert reason == "HIGH_RIAS_OVERRIDE"


def test_tced_row5_detector_unavailable_low():
    """Detector unavailable + RIAS LOW -> INCONCLUSIVE (never defaults to genuine)."""
    engine = DecisionEngine(calibrated=True)
    outcome, reason = engine.decide(
        {"status": "error", "prediction": None},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert outcome != FinalOutcome.GENUINE
    assert reason == "ACOUSTIC_EVIDENCE_UNAVAILABLE"


def test_tced_row6_both_unavailable():
    """Both unavailable -> INCONCLUSIVE (INSUFFICIENT_EVIDENCE)."""
    engine = DecisionEngine(calibrated=True)
    outcome, reason = engine.decide(
        {"status": "error", "prediction": None},
        {"status": "error", "risk_level": None},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert reason == "INSUFFICIENT_EVIDENCE"


def test_tced_row7_invalid_rias_output():
    """Invalid RIAS risk string -> INCONCLUSIVE (INVALID_RIAS_RISK_LEVEL)."""
    engine = DecisionEngine(calibrated=True)
    outcome, reason = engine.decide(
        {"status": "success", "prediction": "GENUINE", "score": 0.3304},
        {"status": "success", "risk_level": "UNRECOGNIZED_LEVEL"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert reason == "INVALID_RIAS_RISK_LEVEL"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
