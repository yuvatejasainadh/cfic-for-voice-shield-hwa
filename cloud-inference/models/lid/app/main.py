"""Language Identification (LID) Model Microservice using Open-Vakgyata.

Model: onecxi/open-vakgyata
Revision: f2754058e485dfc65cc62589b8d0e21c1d328399
License: CC-BY-NC-4.0 (Non-Commercial)
"""

from contextlib import asynccontextmanager
import io
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile, status
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger("lid_service")
logging.basicConfig(level=logging.INFO)

# Configuration & Constants
MODEL_ID = os.getenv("LID_MODEL_ID", "onecxi/open-vakgyata")
MODEL_REVISION = os.getenv("LID_MODEL_REVISION", "f2754058e485dfc65cc62589b8d0e21c1d328399")
WEIGHTS_DIR = os.getenv("LID_WEIGHTS_DIR", "weights")
USE_MOCK = os.getenv("LID_USE_MOCK", "false").lower() == "true"
FORCE_DEVICE = os.getenv("LID_DEVICE", "auto").lower()
TOP_K_DEFAULT = int(os.getenv("LID_TOP_K", "5"))
SILENCE_RMS_THRESHOLD = float(os.getenv("LID_SILENCE_RMS_THRESHOLD", "0.002"))

# Native Open-Vakgyata class labels from config.json
RAW_ID2LABEL: Dict[int, str] = {
    0: "en-IN",
    1: "hi-IN",
    2: "or-IN",
    3: "bn-IN",
    4: "ta-IN",
    5: "te-IN",
    6: "kn-IN",
    7: "ml-IN",
    8: "mr-IN",
    9: "gu-IN",
}

# Normalization to ISO 639-1 project canonical codes
RAW_TO_CANONICAL: Dict[str, str] = {
    "en-IN": "en",
    "hi-IN": "hi",
    "or-IN": "or",
    "bn-IN": "bn",
    "ta-IN": "ta",
    "te-IN": "te",
    "kn-IN": "kn",
    "ml-IN": "ml",
    "mr-IN": "mr",
    "gu-IN": "gu",
}

# In-memory runtime state
_runtime: Dict[str, Any] = {
    "feature_extractor": None,
    "model": None,
    "device": "cpu",
    "loaded": False,
    "error": None,
}


def _is_mock() -> bool:
    return os.getenv("LID_USE_MOCK", "false").lower() == "true"


def _resolve_device() -> str:
    force = os.getenv("LID_DEVICE", "auto").lower()
    if force in ("cpu", "cuda"):
        return force
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model() -> None:
    """Load Open-Vakgyata model and feature extractor."""
    if _is_mock():
        logger.info("LID service configured with USE_MOCK=true; skipping weight loading.")
        _runtime["loaded"] = True
        _runtime["device"] = "mock"
        return

    try:
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        device = _resolve_device()
        _runtime["device"] = device

        # Check if local pre-downloaded weights exist in WEIGHTS_DIR
        load_target = WEIGHTS_DIR if (os.path.exists(WEIGHTS_DIR) and os.listdir(WEIGHTS_DIR)) else MODEL_ID

        logger.info(f"Loading Open-Vakgyata processor from {load_target} (rev: {MODEL_REVISION})...")
        fe = AutoFeatureExtractor.from_pretrained(load_target, revision=MODEL_REVISION)

        logger.info(f"Loading Open-Vakgyata model from {load_target} onto {device}...")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(load_target, revision=MODEL_REVISION)
        model.to(device)
        model.eval()

        _runtime["feature_extractor"] = fe
        _runtime["model"] = model
        _runtime["loaded"] = True
        logger.info(f"Open-Vakgyata LID successfully loaded on {device} (labels: {len(RAW_ID2LABEL)}).")
    except Exception as exc:
        _runtime["loaded"] = False
        _runtime["error"] = str(exc)
        logger.error(f"Failed to load Open-Vakgyata model: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Open-Vakgyata Language Identification (LID) Service",
    version="1.0.0",
    description="Containerized LID microservice serving onecxi/open-vakgyata for Indic & English speech classification",
    lifespan=lifespan,
)


# Pydantic Schemas
class TopPrediction(BaseModel):
    language: str = Field(..., description="Canonical ISO 639-1 code")
    raw_label: str = Field(..., description="Original model label (e.g. en-IN)")
    confidence: float = Field(..., ge=0.0, le=1.0)


class LIDIdentifyResponse(BaseModel):
    status: str = Field(default="success", description="'success', 'silence', or 'error'")
    language: str = Field(..., description="Top predicted canonical language code")
    raw_label: str = Field(..., description="Original model label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Top prediction probability score")
    probabilities: Dict[str, float] = Field(..., description="Canonical language to probability mapping")
    top_k: List[TopPrediction] = Field(default_factory=list)
    model_id: str
    model_revision: str
    latency_ms: float
    audio_duration_seconds: float
    device: str


class LIDHealthResponse(BaseModel):
    status: str
    service: str = "model-lid"
    model_id: str
    model_revision: str
    loaded: bool
    device: str
    is_mock: bool
    error: Optional[str] = None


class LIDMetadataResponse(BaseModel):
    service: str = "model-lid"
    model_id: str
    model_revision: str
    license: str = "CC-BY-NC-4.0"
    device: str
    sampling_rate: int = 16000
    supported_languages: Dict[str, str]
    raw_labels: Dict[int, str]


@app.get("/health", response_model=LIDHealthResponse)
async def health():
    is_mock = _is_mock()
    loaded = _runtime["loaded"] or is_mock
    return LIDHealthResponse(
        status="ok" if loaded else "degraded",
        service="model-lid",
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


@app.get("/metadata", response_model=LIDMetadataResponse)
async def metadata():
    return LIDMetadataResponse(
        service="model-lid",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        license="CC-BY-NC-4.0",
        device=_runtime["device"],
        sampling_rate=16000,
        supported_languages=RAW_TO_CANONICAL,
        raw_labels=RAW_ID2LABEL,
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

    # Silence check: root-mean-square energy
    rms = float(np.sqrt(np.mean(audio**2)))
    is_silent = rms < SILENCE_RMS_THRESHOLD

    return audio, duration, is_silent


@app.post("/identify", response_model=LIDIdentifyResponse)
async def identify(
    file: UploadFile = File(..., description="Audio file (WAV 16kHz mono recommended)"),
    top_k: int = TOP_K_DEFAULT,
):
    start = time.perf_counter()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio payload")

    # Decode and validate
    audio, duration, is_silent = _preprocess_audio(content)

    if is_silent:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        # Handle silence safely: return silence status with zero confidence
        return LIDIdentifyResponse(
            status="silence",
            language="unknown",
            raw_label="silence",
            confidence=0.0,
            probabilities={canon: 0.0 for canon in RAW_TO_CANONICAL.values()},
            top_k=[],
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            latency_ms=elapsed_ms,
            audio_duration_seconds=duration,
            device=_runtime["device"],
        )

    if _is_mock() or not _runtime["loaded"]:
        if _is_mock():
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return LIDIdentifyResponse(
                status="success",
                language="en",
                raw_label="en-IN",
                confidence=0.99,
                probabilities={"en": 0.99, "hi": 0.005, "te": 0.005},
                top_k=[TopPrediction(language="en", raw_label="en-IN", confidence=0.99)],
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                latency_ms=elapsed_ms,
                audio_duration_seconds=duration,
                device="mock",
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LID model not loaded: {_runtime.get('error')}",
        )

    import torch
    fe = _runtime["feature_extractor"]
    model = _runtime["model"]
    device = _runtime["device"]

    try:
        inputs = fe(audio, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LID inference error: {exc}",
        )

    probs_cpu = probs.detach().cpu().numpy()
    
    # Construct canonical probabilities dictionary
    canonical_probs: Dict[str, float] = {}
    top_candidates: List[tuple[str, str, float]] = []

    for idx, raw_label in RAW_ID2LABEL.items():
        p_val = float(probs_cpu[idx])
        canon_lang = RAW_TO_CANONICAL.get(raw_label, "unknown")
        # In case multiple map to same canonical, take max
        canonical_probs[canon_lang] = max(canonical_probs.get(canon_lang, 0.0), round(p_val, 6))
        top_candidates.append((canon_lang, raw_label, p_val))

    # Rank by confidence descending
    top_candidates.sort(key=lambda x: x[2], reverse=True)
    top_idx = int(np.argmax(probs_cpu))
    top_raw = RAW_ID2LABEL[top_idx]
    top_canon = RAW_TO_CANONICAL.get(top_raw, "unknown")
    top_conf = round(float(probs_cpu[top_idx]), 6)

    k_items = [
        TopPrediction(language=c, raw_label=r, confidence=round(p, 6))
        for c, r, p in top_candidates[: min(top_k, len(top_candidates))]
    ]

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return LIDIdentifyResponse(
        status="success",
        language=top_canon,
        raw_label=top_raw,
        confidence=top_conf,
        probabilities=canonical_probs,
        top_k=k_items,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        latency_ms=elapsed_ms,
        audio_duration_seconds=duration,
        device=device,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8301)
