"""Unit tests for Whisper English ASR microservice."""

import io
import os
import sys
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

os.environ["ASR_EN_USE_MOCK"] = "true"

# Insert app directory into sys.path
ASR_APP_DIR = str(Path(__file__).resolve().parents[1] / "app")
if ASR_APP_DIR not in sys.path:
    sys.path.insert(0, ASR_APP_DIR)

import main as asr_main


@pytest.fixture
def client():
    return TestClient(asr_main.app)


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


def test_asr_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "model-asr-en"
    assert data["target_language"] == "en"
    assert data["model_id"] == "openai/whisper-base.en"
    assert data["model_revision"] == "911407f4214e0e1d82085af863093ec0b66f9cd6"


def test_asr_metadata_endpoint(client):
    res = client.get("/metadata")
    assert res.status_code == 200
    data = res.json()
    assert data["license"] == "Apache-2.0"
    assert data["target_language"] == "en"


def test_asr_transcribe_mock(client):
    wav_bytes = _generate_wav_bytes(duration_s=2.0)
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    res = client.post("/transcribe", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["language"] == "en"
    assert data["text"] is not None
    assert len(data["text"]) > 0


def test_asr_silence_handling(client):
    silence_bytes = _generate_silence_bytes(duration_s=2.0)
    files = {"file": ("silence.wav", silence_bytes, "audio/wav")}
    res = client.post("/transcribe", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "silence"
    assert data["text"] == ""


def test_asr_empty_payload(client):
    files = {"file": ("empty.wav", b"", "audio/wav")}
    res = client.post("/transcribe", files=files)
    assert res.status_code == 400
