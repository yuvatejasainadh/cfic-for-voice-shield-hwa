"""FastAPI Microservice for PellaV2 K12 INT8 Acoustic Deepfake Detection."""

import hashlib
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, status
import numpy as np
import soundfile as sf
import torch

try:
    from app.config import settings
    from app.model import PellaV2Model, extract_160_log_mel_features, load_pellav2_checkpoint
    from app.schemas import (
        ReadinessResponse,
        SpoofHealthResponse,
        SpoofMetadataResponse,
        SpoofPredictionResponse,
    )
except ImportError:
    from config import settings
    from model import PellaV2Model, extract_160_log_mel_features, load_pellav2_checkpoint
    from schemas import (
        ReadinessResponse,
        SpoofHealthResponse,
        SpoofMetadataResponse,
        SpoofPredictionResponse,
    )

logger = logging.getLogger("pellav2_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")

_runtime: Dict[str, Any] = {
    "model": None,
    "loaded": False,
    "is_mock": False,
    "device": "cpu",
    "error": None,
    "artifact_found": False,
    "checksum_verified": False,
}


def _verify_checksum(path: str, expected_sha256: str) -> bool:
    """Verify SHA-256 hash of the model checkpoint."""
    if not os.path.exists(path):
        return False
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        actual = h.hexdigest().lower()
        matched = actual == expected_sha256.lower()
        if not matched:
            logger.warning(f"Checksum mismatch for {path}: actual={actual} vs expected={expected_sha256}")
        return matched
    except Exception as exc:
        logger.error(f"Failed to compute checksum for {path}: {exc}")
        return False


def _load_runtime_model():
    """Load model once per worker lifecycle."""
    _runtime["is_mock"] = settings.use_mock
    _runtime["device"] = settings.device
    _runtime["artifact_found"] = os.path.exists(settings.model_path)

    if settings.use_mock:
        logger.info("PELLAV2_USE_MOCK=true: Initializing mock detection runtime.")
        _runtime["loaded"] = True
        return

    if not _runtime["artifact_found"]:
        err_msg = f"Model artifact not found at {settings.model_path}."
        logger.error(err_msg)
        _runtime["loaded"] = False
        _runtime["error"] = err_msg
        return

    # Verify SHA-256 integrity
    logger.info(f"Verifying integrity of checkpoint {settings.model_path}...")
    _runtime["checksum_verified"] = _verify_checksum(settings.model_path, settings.expected_sha256)
    if not _runtime["checksum_verified"]:
        logger.warning("Checksum verification failed or checksum metadata unverified. Loading safely with inspection.")

    try:
        logger.info(f"Loading PellaV2 K12 INT8 weights from {settings.model_path}...")
        t0 = time.perf_counter()
        model = load_pellav2_checkpoint(settings.model_path, settings.calibration_path)
        load_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        _runtime["model"] = model
        _runtime["loaded"] = True
        _runtime["error"] = None
        logger.info(f"PellaV2 model ready for inference (cold load: {load_time_ms} ms).")
    except Exception as exc:
        _runtime["loaded"] = False
        _runtime["error"] = str(exc)
        logger.error(f"Failed to load PellaV2 checkpoint: {exc}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown."""
    logger.info(f"Starting {settings.service_name} (port {settings.port})...")
    _load_runtime_model()
    yield
    logger.info(f"Shutting down {settings.service_name}...")
    _runtime["model"] = None
    _runtime["loaded"] = False


app = FastAPI(
    title="PellaV2 K12 INT8 Acoustic Deepfake Detection Service",
    version="1.0.0",
    description="Containerized INT8 dynamic quantized acoustic deepfake detection service for CFIC x VoiceShield",
    lifespan=lifespan,
)


@app.get("/health", response_model=SpoofHealthResponse, tags=["System"])
async def health():
    """Liveness probe reporting model status and artifact presence."""
    return SpoofHealthResponse(
        status="ok",
        service=settings.service_name,
        architecture="Conformer-K12-INT8",
        target_domain="Acoustic-Synthetic-Detection",
        model_version=settings.model_version,
        artifact_found=os.path.exists(settings.model_path),
        checksum_verified=_runtime["checksum_verified"],
        is_mock=_runtime["is_mock"],
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["System"])
async def ready():
    """Readiness probe indicating if the service is ready for traffic."""
    if not _runtime["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PellaV2 model is not ready: {_runtime.get('error', 'uninitialized')}",
        )
    return ReadinessResponse(
        status="ready",
        ready=True,
        model_loaded=_runtime["model"] is not None or _runtime["is_mock"],
    )


@app.get("/metadata", response_model=SpoofMetadataResponse, tags=["System"])
async def metadata():
    """Model card metadata and calibrated configuration."""
    cal_params = {}
    if _runtime["model"] is not None:
        cal_params = {"a": _runtime["model"].cal_a, "b": _runtime["model"].cal_b}

    return SpoofMetadataResponse(
        service=settings.service_name,
        model_id="varunpadavala/cfic-pellav2-k12-int8",
        model_version=settings.model_version,
        architecture="Conformer-K12-1024",
        quantization="INT8-Dynamic-Affine",
        sample_rate=settings.sample_rate,
        feature_dim=160,
        threshold=settings.threshold,
        calibration_parameters=cal_params,
    )


@app.post("/predict", response_model=SpoofPredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(..., description="Audio payload (16 kHz mono WAV recommended)")):
    """Run acoustic deepfake detection on the uploaded audio payload."""
    start = time.perf_counter()
    content = await file.read()

    # Audio input validation
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio payload")

    if len(content) > settings.max_audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio payload exceeds {settings.max_audio_bytes // (1024 * 1024)} MiB limit",
        )

    # Mock mode fallback for scaffold testing
    if _runtime["is_mock"]:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return SpoofPredictionResponse(
            status="success",
            prediction="GENUINE",
            score=0.08,
            threshold=settings.threshold,
            model_version=f"{settings.model_version}-mock",
            latency_ms=elapsed_ms,
        )

    # Check if model is loaded
    if not _runtime["loaded"] or _runtime["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PellaV2 model unavailable: {_runtime.get('error', 'Weights not loaded')}",
        )

    # Decode audio bytes safely
    try:
        audio_stream = io.BytesIO(content)
        wav, sr = sf.read(audio_stream, dtype="float32")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or unsupported audio encoding: {exc}",
        )

    if wav is None or wav.size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Decoded audio is empty")

    if np.isnan(wav).any() or np.isinf(wav).any():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio contains NaN or Inf values")

    # Multi-channel to mono
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)

    # Feature extraction & Model Forward Pass
    try:
        features = extract_160_log_mel_features(
            audio=wav,
            sample_rate=sr,
            target_sample_rate=settings.sample_rate,
            n_mels=160,
        )

        with torch.no_grad():
            logit, calibrated_score = _runtime["model"](features)

        score_val = float(calibrated_score.item())
        # Enforce valid probability bounds [0.0, 1.0]
        score_val = max(0.0, min(1.0, score_val))

        # Prediction polarity: score >= threshold indicates synthetic/fake voice
        verdict = "FAKE" if score_val >= settings.threshold else "GENUINE"
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        return SpoofPredictionResponse(
            status="success",
            prediction=verdict,
            score=round(score_val, 4),
            threshold=settings.threshold,
            model_version=settings.model_version,
            latency_ms=elapsed_ms,
        )
    except Exception as exc:
        logger.error(f"Inference exception during forward pass: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {type(exc).__name__}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
