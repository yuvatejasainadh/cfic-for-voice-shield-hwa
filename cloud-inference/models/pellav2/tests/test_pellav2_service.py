"""Unit and service contract tests for PellaV2 K12 INT8 Detection Microservice."""

import io
import math
import os
from pathlib import Path
import sys
import wave
import pytest
from fastapi.testclient import TestClient

# Ensure pellav2 app directory is at the head of sys.path
PELLAV2_APP_DIR = str(Path(__file__).resolve().parents[1] / "app")
PELLAV2_ROOT_DIR = str(Path(__file__).resolve().parents[1])
if PELLAV2_APP_DIR not in sys.path:
    sys.path.insert(0, PELLAV2_APP_DIR)

import main as pella_main
from config import settings
from model import extract_160_log_mel_features

REAL_WAV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../cfic-asr-benchmark/data/audio/telugu/telugu_00000.wav")
)
CHECKPOINT_PATH = os.path.abspath(
    os.path.join(PELLAV2_ROOT_DIR, "weights/pellav2_k12_int8_final.pt")
)
CALIBRATION_PATH = os.path.abspath(
    os.path.join(PELLAV2_ROOT_DIR, "config/k12_calibration.json")
)


def create_dummy_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate in-memory valid PCM 16-bit mono WAV bytes."""
    num_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = bytearray()
        for i in range(num_samples):
            val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            samples.extend(val.to_bytes(2, byteorder="little", signed=True))
        wf.writeframes(samples)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    # Configure real model path for test suite if checkpoint exists
    if os.path.exists(CHECKPOINT_PATH):
        settings.model_path = CHECKPOINT_PATH
        settings.calibration_path = CALIBRATION_PATH
        settings.use_mock = False
    else:
        settings.use_mock = True

    with TestClient(pella_main.app) as test_client:
        yield test_client


class TestPellaV2Service:
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "model-pellav2"
        assert data["model_version"] == "pellav2-k12-int8"
        assert "artifact_found" in data
        assert "checksum_verified" in data

    def test_ready_endpoint(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["ready"] is True
        assert data["model_loaded"] is True

    def test_metadata_endpoint(self, client):
        resp = client.get("/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "model-pellav2"
        assert data["architecture"] == "Conformer-K12-1024"
        assert data["quantization"] == "INT8-Dynamic-Affine"
        assert data["feature_dim"] == 160
        assert data["threshold"] == 0.50
        assert "calibration_parameters" in data

    def test_empty_audio_rejected(self, client):
        files = {"file": ("empty.wav", b"", "audio/wav")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 400
        assert "Empty audio payload" in resp.json()["detail"]

    def test_corrupted_audio_rejected(self, client):
        files = {"file": ("corrupted.wav", b"NOT_VALID_AUDIO_BYTES_HEADER", "audio/wav")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 400
        assert "Corrupted or unsupported" in resp.json()["detail"]

    def test_oversized_audio_rejected(self, client, monkeypatch):
        monkeypatch.setattr(settings, "max_audio_bytes", 1024)
        files = {"file": ("oversized.wav", b"x" * 2048, "audio/wav")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 413
        assert "limit" in resp.json()["detail"]

    def test_mock_prediction(self, client, monkeypatch):
        monkeypatch.setitem(pella_main._runtime, "is_mock", True)
        wav_bytes = create_dummy_wav_bytes(duration_s=1.0)
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["prediction"] == "GENUINE"
        assert data["score"] == 0.08
        assert "mock" in data["model_version"]
        monkeypatch.setitem(pella_main._runtime, "is_mock", False)

    def test_real_audio_inference(self, client):
        if not os.path.exists(REAL_WAV_PATH):
            pytest.skip(f"Real test audio not found at {REAL_WAV_PATH}")

        with open(REAL_WAV_PATH, "rb") as f:
            wav_bytes = f.read()

        files = {"file": ("telugu_real.wav", wav_bytes, "audio/wav")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["prediction"] in ("GENUINE", "FAKE")
        assert 0.0 <= data["score"] <= 1.0
        assert data["threshold"] == 0.50
        assert data["model_version"] == "pellav2-k12-int8"
        assert data["latency_ms"] > 0.0

    def test_repeated_inference_stability(self, client):
        """Verify model weights are loaded once and state is maintained across multiple requests."""
        initial_model = pella_main._runtime["model"]
        wav_bytes = create_dummy_wav_bytes(duration_s=1.0)
        files = {"file": ("test1.wav", wav_bytes, "audio/wav")}

        r1 = client.post("/predict", files=files)
        r2 = client.post("/predict", files=files)

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert pella_main._runtime["model"] is initial_model

    def test_model_unloaded_returns_503(self, client, monkeypatch):
        monkeypatch.setitem(pella_main._runtime, "loaded", False)
        monkeypatch.setitem(pella_main._runtime, "model", None)
        monkeypatch.setitem(pella_main._runtime, "is_mock", False)

        wav_bytes = create_dummy_wav_bytes(duration_s=1.0)
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        resp = client.post("/predict", files=files)
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

    def test_feature_extraction_dimension(self):
        import numpy as np
        fake_audio = np.random.uniform(-0.5, 0.5, 16000 * 2).astype(np.float32)
        feats = extract_160_log_mel_features(fake_audio, sample_rate=16000)
        assert feats.ndim == 3
        assert feats.shape[0] == 1
        assert feats.shape[2] == 160
