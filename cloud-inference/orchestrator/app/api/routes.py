"""API routes for Cloud Inference Orchestrator."""

import asyncio
import time
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.core.config import settings
from app.routing.language_router import language_router
from app.routing.spoof_selector import spoof_selector
from app.schemas.inference import (
    ASRResult,
    HealthResponse,
    InferenceResponse,
    LIDResult,
    SpoofResult,
)
from app.services.asr_client import asr_client
from app.services.lid_client import lid_client
from app.services.spoof_client import spoof_client

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version="0.1.0",
        backends={
            "lid": settings.lid_service_url,
            "asr_en": settings.asr_en_service_url,
            "asr_te": settings.asr_te_service_url,
            "asr_hi": settings.asr_hi_service_url,
            "vmc_south": settings.vmc_south_service_url,
            "aasist_fallback": settings.aasist_fallback_service_url,
        },
    )


@router.post(
    "/v1/inference",
    response_model=InferenceResponse,
    tags=["Inference"],
    summary="Run full model orchestration: LID -> Routed ASR + Routed Spoof Detection",
)
async def run_inference(
    file: UploadFile = File(..., description="Audio file payload for model evaluation"),
):
    start = time.perf_counter()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    filename = file.filename or "audio.wav"
    content_type = file.content_type or "audio/wav"

    # Step 1: Language Identification
    lid_res = await lid_client.identify(content, filename=filename, content_type=content_type)
    detected_lang = lid_res.get("language", "en")

    # Step 2: Model Routing
    asr_endpoint, normalized_lang = language_router.select_asr_endpoint(detected_lang)
    spoof_endpoint, spoof_model = spoof_selector.select_spoof_endpoint(detected_lang)

    # Step 3: Concurrent execution of ASR and Acoustic Spoof Detection
    asr_task = asyncio.create_task(
        asr_client.transcribe(asr_endpoint, content, filename=filename, content_type=content_type)
    )
    spoof_task = asyncio.create_task(
        spoof_client.predict(spoof_endpoint, content, filename=filename, content_type=content_type)
    )

    asr_raw, spoof_raw = await asyncio.gather(asr_task, spoof_task, return_exceptions=True)

    # Normalize ASR result
    if isinstance(asr_raw, Exception):
        asr_data = {
            "status": "error",
            "text": None,
            "language": normalized_lang,
            "model_version": "indicconformer-error",
            "latency_ms": 0.0,
        }
    else:
        asr_data = asr_raw

    # Normalize Spoof result
    if isinstance(spoof_raw, Exception):
        spoof_data = {
            "status": "error",
            "prediction": None,
            "score": None,
            "model_version": spoof_model,
            "latency_ms": 0.0,
        }
    else:
        spoof_data = spoof_raw

    total_latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return InferenceResponse(
        status="success",
        lid=LIDResult(
            status=lid_res.get("status", "success"),
            language=detected_lang,
            confidence=lid_res.get("confidence", 1.0),
            model_version=lid_res.get("model_version", "lid-v0.1"),
            latency_ms=lid_res.get("latency_ms", 0.0),
        ),
        transcription=ASRResult(
            status=asr_data.get("status", "success"),
            text=asr_data.get("text"),
            language=asr_data.get("language", normalized_lang),
            model_version=asr_data.get("model_version", "indicconformer-v0.1"),
            latency_ms=asr_data.get("latency_ms", 0.0),
        ),
        spoof=SpoofResult(
            status=spoof_data.get("status", "success"),
            prediction=spoof_data.get("prediction"),
            score=spoof_data.get("score"),
            model_version=spoof_data.get("model_version", spoof_model),
            latency_ms=spoof_data.get("latency_ms", 0.0),
        ),
        latency_ms=total_latency_ms,
    )
