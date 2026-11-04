"""Unit tests for Open-Vakgyata LID microservice."""

import io
import os
import sys
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

os.environ["LID_USE_MOCK"] = "true"

# Insert app directory into sys.path
LID_APP_DIR = str(Path(__file__).resolve().parents[1] / "app")
if LID_APP_DIR not in sys.path:
    sys.path.insert(0, LID_APP_DIR)

import main as lid_main


@pytest.fixture
def client():
    return TestClient(lid_main.app)


def _generate_wav_bytes(duration_s: float = 1.0, sr: int = 16000, freq: float = 440.0) -> bytes:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    return buf.getvalue()


def _generate_silence_bytes(duration_s: float = 1.0, sr: int = 16000) -> bytes:
    waveform = np.zeros(int(sr * duration_s), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    return buf.getvalue()


def test_lid_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "model-lid"
    assert data["model_id"] == "onecxi/open-vakgyata"
    assert data["model_revision"] == "f2754058e485dfc65cc62589b8d0e21c1d328399"


def test_lid_metadata_endpoint(client):
    res = client.get("/metadata")
    assert res.status_code == 200
    data = res.json()
    assert data["license"] == "CC-BY-NC-4.0"
    assert "en" in data["supported_languages"].values()
    assert "te" in data["supported_languages"].values()
    assert "hi" in data["supported_languages"].values()
    assert len(data["raw_labels"]) == 10


def test_lid_identify_mock(client):
    wav_bytes = _generate_wav_bytes(duration_s=2.0)
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    res = client.post("/identify?top_k=3", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["language"] == "en"
    assert data["raw_label"] == "en-IN"
    assert data["confidence"] > 0.90
    assert len(data["top_k"]) <= 3


def test_lid_silence_handling(client):
    silence_bytes = _generate_silence_bytes(duration_s=2.0)
    files = {"file": ("silence.wav", silence_bytes, "audio/wav")}
    res = client.post("/identify", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "silence"
    assert data["language"] == "unknown"
    assert data["confidence"] == 0.0


def test_lid_empty_payload(client):
    files = {"file": ("empty.wav", b"", "audio/wav")}
    res = client.post("/identify", files=files)
    assert res.status_code == 400
