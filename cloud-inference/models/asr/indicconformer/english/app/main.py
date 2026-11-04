"""Dedicated English Automatic Speech Recognition (ASR) Microservice using Whisper Base English.

Model: openai/whisper-base.en
Revision: 911407f4214e0e1d82085af863093ec0b66f9cd6
License: Apache-2.0
"""

from contextlib import asynccontextmanager
import io
import logging
import math
import os
import time
from typing import Any, Dict, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile, status
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger("english_asr_service")
logging.basicConfig(level=logging.INFO)

# Configuration & Constants
MODEL_ID = os.getenv("ASR_EN_MODEL_ID", "openai/whisper-base.en")
MODEL_REVISION = os.getenv("ASR_EN_MODEL_REVISION", "911407f4214e0e1d82085af863093ec0b66f9cd6")
WEIGHTS_DIR = os.getenv("ASR_EN_WEIGHTS_DIR", "weights")
USE_MOCK = os.getenv("ASR_EN_USE_MOCK", "false").lower() == "true"
FORCE_DEVICE = os.getenv("ASR_EN_DEVICE", "auto").lower()
SILENCE_RMS_THRESHOLD = float(os.getenv("ASR_EN_SILENCE_RMS_THRESHOLD", "0.002"))
MAX_NEW_TOKENS = int(os.getenv("ASR_EN_MAX_NEW_TOKENS", "448"))
BEAM_SIZE = int(os.getenv("ASR_EN_BEAM_SIZE", "1"))

# In-memory runtime state
_runtime: Dict[str, Any] = {
    "processor": None,
    "model": None,
    "device": "cpu",
    "loaded": False,
    "error": None,
}


def _is_mock() -> bool:
    return os.getenv("ASR_EN_USE_MOCK", "false").lower() == "true"


def _resolve_device() -> str:
    force = os.getenv("ASR_EN_DEVICE", "auto").lower()
    if force in ("cpu", "cuda"):
        return force
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model() -> None:
    """Load Whisper English-only model and processor."""
    if _is_mock():
        logger.info("English ASR service configured with USE_MOCK=true; skipping weight loading.")
        _runtime["loaded"] = True
        _runtime["device"] = "mock"
        return

    try:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        device = _resolve_device()
        _runtime["device"] = device

        load_target = WEIGHTS_DIR if (os.path.exists(WEIGHTS_DIR) and os.listdir(WEIGHTS_DIR)) else MODEL_ID

        logger.info(f"Loading Whisper processor from {load_target} (rev: {MODEL_REVISION})...")
        processor = WhisperProcessor.from_pretrained(load_target, revision=MODEL_REVISION)

        logger.info(f"Loading Whisper model from {load_target} onto {device}...")
        model = WhisperForConditionalGeneration.from_pretrained(load_target, revision=MODEL_REVISION)
        model.to(device)
        model.eval()

        _runtime["processor"] = processor
        _runtime["model"] = model
        _runtime["loaded"] = True
        logger.info(f"Whisper English ASR successfully loaded on {device}.")
    except Exception as exc:
        _runtime["loaded"] = False
        _runtime["error"] = str(exc)
        logger.error(f"Failed to load Whisper English ASR model: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="English ASR Service (openai/whisper-base.en)",
    version="1.0.0",
    description="Containerized English speech-to-text service serving openai/whisper-base.en",
    lifespan=lifespan,
)


class ASRResponse(BaseModel):
    status: str = Field(default="success", description="'success', 'silence', or 'error'")
    text: Optional[str] = Field(default=None, description="Decoded transcription text")
    language: str = Field(default="en", description="Target language code")
    model_id: str
    model_revision: str
    latency_ms: float
    audio_duration_seconds: float
    device: str


class ASRHealthResponse(BaseModel):
    status: str
    service: str = "model-asr-en"
    target_language: str = "en"
    model_id: str
    model_revision: str
    loaded: bool
    device: str
    is_mock: bool
    error: Optional[str] = None


class ASRMetadataResponse(BaseModel):
    service: str = "model-asr-en"
    target_language: str = "en"
    model_id: str
    model_revision: str
    license: str = "Apache-2.0"
    device: str
    sampling_rate: int = 16000
    architecture: str = "WhisperForConditionalGeneration"


@app.get("/health", response_model=ASRHealthResponse)
async def health():
    is_mock = _is_mock()
    loaded = _runtime["loaded"] or is_mock
    return ASRHealthResponse(
        status="ok" if loaded else "degraded",
        service="model-asr-en",
        target_language="en",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        loaded=loaded,
        device="mock" if is_mock else _runtime["device"],
        is_mock=is_mock,
        error=_runtime.get("error"),
    )


@app.get("/ready")
async def ready():
    if not _runtime["loaded"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready")
    return {"status": "ready"}


@app.get("/metadata", response_model=ASRMetadataResponse)
async def metadata():
    return ASRMetadataResponse(
        service="model-asr-en",
        target_language="en",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        license="Apache-2.0",
        device=_runtime["device"],
        sampling_rate=16000,
        architecture="WhisperForConditionalGeneration",
    )


def _preprocess_audio(raw_bytes: bytes) -> tuple[np.ndarray, float, bool]:
    """Decode audio bytes to 16 kHz mono float32 waveform, validate, and check silence."""
    import soundfile as sf
    try:
        audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed audio payload: {exc}",
        )

    if audio is None or audio.size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio payload")

    # Multi-channel to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    # Resample to 16 kHz if necessary
    if sr != 16000:
        from scipy.signal import resample_poly
        gcd = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // gcd, sr // gcd).astype(np.float32)

    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    duration = float(len(audio) / 16000.0)

    # Silence check
    rms = float(np.sqrt(np.mean(audio**2)))
    is_silent = rms < SILENCE_RMS_THRESHOLD

    return audio, duration, is_silent


@app.post("/transcribe", response_model=ASRResponse)
async def transcribe(file: UploadFile = File(..., description="16 kHz mono WAV audio")):
    start = time.perf_counter()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio payload")

    audio, duration, is_silent = _preprocess_audio(content)

    if is_silent:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return ASRResponse(
            status="silence",
            text="",
            language="en",
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            latency_ms=elapsed_ms,
            audio_duration_seconds=duration,
            device=_runtime["device"],
        )

    if _is_mock() or not _runtime["loaded"]:
        if _is_mock():
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return ASRResponse(
                status="success",
                text="Please verify your urgent banking request immediately.",
                language="en",
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                latency_ms=elapsed_ms,
                audio_duration_seconds=duration,
                device="mock",
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"English ASR model not loaded: {_runtime.get('error')}",
        )

    import torch
    processor = _runtime["processor"]
    model = _runtime["model"]
    device = _runtime["device"]

    try:
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.inference_mode():
            if BEAM_SIZE > 1:
                predicted_ids = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS, num_beams=BEAM_SIZE)
            else:
                predicted_ids = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        clean_text = transcription.strip()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Whisper transcription error: {exc}",
        )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return ASRResponse(
        status="success",
        text=clean_text,
        language="en",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        latency_ms=elapsed_ms,
        audio_duration_seconds=duration,
        device=device,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8311)
