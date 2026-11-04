"""API routes for Cloud Inference Orchestrator."""

import asyncio
import time
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
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
        version="0.2.0",
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
    summary="Run full model orchestration: Vakgyata LID -> Routed ASR + Routed Spoof Detection",
)
async def run_inference(
    file: UploadFile = File(..., description="Audio file payload (10-second RIAS chunk recommended)"),
    session_id: Optional[str] = Form(default=None, description="Optional tracking session ID"),
    chunk_index: Optional[int] = Form(default=0, description="Sequential index of audio chunk"),
    timestamp_start_s: Optional[float] = Form(default=0.0, description="Chunk start offset in seconds"),
    timestamp_end_s: Optional[float] = Form(default=10.0, description="Chunk end offset in seconds"),
):
    start = time.perf_counter()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio payload")

    filename = file.filename or "chunk.wav"
    content_type = file.content_type or "audio/wav"

    # Step 1: Language Identification with Vakgyata (with retries)
    lid_res = await lid_client.identify(content, filename=filename, content_type=content_type)
    detected_lang = lid_res.get("language", "unknown")
    raw_label = lid_res.get("raw_label")
    lid_conf = float(lid_res.get("confidence", 0.0))
    lid_status = lid_res.get("status", "error")

    # Evaluate confidence threshold
    is_lid_confident = (
        lid_status == "success"
        and detected_lang != "unknown"
        and lid_conf >= settings.lid_confidence_threshold
    )

    if not is_lid_confident:
        # Mark LID as inconclusive if it ran successfully but failed threshold
        if lid_status == "success":
            lid_status = "inconclusive"

    lid_result = LIDResult(
        status=lid_status,
        language=detected_lang,
        raw_label=raw_label,
        confidence=round(lid_conf, 4),
        probabilities=lid_res.get("probabilities"),
        top_k=lid_res.get("top_k"),
        model_id=lid_res.get("model_id", "onecxi/open-vakgyata"),
        model_revision=lid_res.get("model_revision"),
        retries_used=lid_res.get("retries_used", 0),
        latency_ms=lid_res.get("latency_ms", 0.0),
    )

    # Step 2: Spoof Detection (runs regardless or concurrently)
    spoof_endpoint, spoof_model = spoof_selector.select_spoof_endpoint(detected_lang)
    spoof_task = asyncio.create_task(
        spoof_client.predict(spoof_endpoint, content, filename=filename, content_type=content_type)
    )

    # Step 3: ASR Routing Policy
    if not is_lid_confident:
        # Rule: Do NOT call ASR when LID is inconclusive
        asr_result = ASRResult(
            status="skipped",
            text=None,
            language=detected_lang,
            model_id=None,
            model_revision=None,
            latency_ms=0.0,
        )
        pipeline_status = "inconclusive"
        spoof_raw = await spoof_task
    else:
        # Route to exact language-specific ASR backend
        asr_endpoint, normalized_lang = language_router.select_asr_endpoint(detected_lang)
        asr_task = asyncio.create_task(
            asr_client.transcribe(asr_endpoint, content, filename=filename, content_type=content_type)
        )

        asr_raw, spoof_raw = await asyncio.gather(asr_task, spoof_task, return_exceptions=True)

        if isinstance(asr_raw, Exception) or asr_raw.get("status") == "error":
            asr_result = ASRResult(
                status="error",
                text=None,
                language=normalized_lang,
                model_id=asr_raw.get("model_id") if isinstance(asr_raw, dict) else None,
                model_revision=asr_raw.get("model_revision") if isinstance(asr_raw, dict) else None,
                latency_ms=asr_raw.get("latency_ms", 0.0) if isinstance(asr_raw, dict) else 0.0,
            )
            # ASR failure produces INCONCLUSIVE overall
            pipeline_status = "inconclusive"
        else:
            asr_result = ASRResult(
                status=asr_raw.get("status", "success"),
                text=asr_raw.get("text"),
                language=asr_raw.get("language", normalized_lang),
                model_id=asr_raw.get("model_id", "openai/whisper-base.en" if normalized_lang == "en" else "indicconformer"),
                model_revision=asr_raw.get("model_revision"),
                latency_ms=asr_raw.get("latency_ms", 0.0),
            )
            pipeline_status = "success"

    # Normalize Spoof result
    if isinstance(spoof_raw, Exception) or (isinstance(spoof_raw, dict) and spoof_raw.get("status") == "error"):
        spoof_result = SpoofResult(
            status="error",
            prediction=None,
            score=None,
            model_version=spoof_model,
            latency_ms=0.0,
        )
    else:
        spoof_result = SpoofResult(
            status=spoof_raw.get("status", "success"),
            prediction=spoof_raw.get("prediction"),
            score=spoof_raw.get("score"),
            model_version=spoof_raw.get("model_version", spoof_model),
            latency_ms=spoof_raw.get("latency_ms", 0.0),
        )

    total_latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return InferenceResponse(
        status=pipeline_status,
        session_id=session_id,
        chunk_index=chunk_index,
        timestamp_start_s=timestamp_start_s,
        timestamp_end_s=timestamp_end_s,
        lid=lid_result,
        transcription=asr_result,
        spoof=spoof_result,
        latency_ms=total_latency_ms,
    )
