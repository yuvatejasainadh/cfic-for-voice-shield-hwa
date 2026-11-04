"""Analysis Orchestrator coordinating audio ingestion, inference services, and decision fusion."""

import asyncio
import time
import uuid
from typing import Any
from app.orchestration.decision_engine import decide
from app.schemas.analysis import AnalysisResponse, RiskLevel
from app.services.audio_service import audio_service
from app.services.cfic_client import (
    CFICInferenceClient,
    LegacyPellaClient,
    LegacyTranscriptionClient,
    cfic_client,
)
from app.services.job_service import JobStatus, job_service
from app.services.risk_service import risk_service


class AnalysisOrchestrator:
    """Orchestrates application-level execution of deepfake detection and semantic risk analysis."""

    def __init__(
        self,
        pella_client: Any | None = None,
        transcription_client: Any | None = None,
        cloud_client: CFICInferenceClient | None = None,
    ):
        # Support legacy test injection or default legacy clients
        self.pella_client = pella_client or LegacyPellaClient()
        self.transcription_client = transcription_client or LegacyTranscriptionClient()
        self.cloud_client = cloud_client or cfic_client

    async def analyze_audio(
        self,
        audio: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        request_id: str | None = None,
    ) -> AnalysisResponse:
        """Run complete audio analysis pipeline."""
        req_id = request_id or str(uuid.uuid4())
        pipeline_start = time.perf_counter()
        latencies: dict[str, float] = {}

        # Track job in application job service
        job_service.create_job(request_id=req_id, metadata={"filename": filename})
        job_service.update_status(req_id, JobStatus.PROCESSING)

        # Stage 1: Parallel dispatch of acoustic detector and transcription
        pella_task = asyncio.create_task(
            self.pella_client.predict(audio, filename, content_type)
        )
        transcription_task = asyncio.create_task(
            self.transcription_client.transcribe(audio, filename, content_type)
        )

        pella_res_raw, transc_res_raw = await asyncio.gather(
            pella_task, transcription_task, return_exceptions=True
        )

        # Normalize acoustic detector outcome
        if isinstance(pella_res_raw, Exception):
            pella_res: dict[str, Any] = {
                "status": "error",
                "prediction": None,
                "score": None,
                "model_version": None,
                "error": str(pella_res_raw),
                "latency_ms": 0.0,
            }
        else:
            pella_res = pella_res_raw

        latencies["pella"] = pella_res.get("latency_ms", 0.0)

        # Normalize transcription outcome
        if isinstance(transc_res_raw, Exception):
            transc_res: dict[str, Any] = {
                "status": "error",
                "text": None,
                "language": None,
                "model_version": None,
                "error": str(transc_res_raw),
                "latency_ms": 0.0,
            }
        else:
            transc_res = transc_res_raw

        latencies["transcription"] = transc_res.get("latency_ms", 0.0)

        # Stage 2: Semantic Risk Assessment via RiskService (triggered if transcription produced text)
        rias_res: dict[str, Any] = {
            "status": "skipped",
            "risk_level": None,
            "reason": "Transcription was unsuccessful or produced no text",
            "latency_ms": 0.0,
        }

        if transc_res.get("status") == "success" and transc_res.get("text"):
            transcript_text = transc_res["text"]
            try:
                import app.orchestration.service as orch_service_module
                assess_fn = getattr(orch_service_module, "assess_transcript", risk_service.assess_transcript)
                rias_res = await assess_fn(transcript_text)
            except Exception as exc:
                rias_res = {
                    "status": "error",
                    "risk_level": None,
                    "reason": None,
                    "error": f"LLM assessment failed: {type(exc).__name__}",
                    "latency_ms": 0.0,
                }

        latencies["rias_llm"] = rias_res.get("latency_ms", 0.0)

        # Stage 3: Decision Engine Synthesis
        outcome, reason = decide(pella_res, rias_res)

        total_latency = round((time.perf_counter() - pipeline_start) * 1000, 2)
        latencies["total"] = total_latency

        risk_level_enum = None
        if rias_res.get("risk_level"):
            try:
                risk_level_enum = RiskLevel(rias_res["risk_level"])
            except ValueError:
                risk_level_enum = None

        response = AnalysisResponse(
            request_id=req_id,
            final_outcome=outcome,
            reason_code=reason,
            pella_status=pella_res.get("status", "error"),
            rias_transcription_status=transc_res.get("status", "error"),
            rias_risk_status=rias_res.get("status", "error"),
            rias_risk_level=risk_level_enum,
            model_version=pella_res.get("model_version"),
            latency_ms=latencies,
        )

        job_service.complete_job(req_id, response.model_dump())
        return response


# Global default instance & legacy aliases
orchestrator = AnalysisOrchestrator()
PipelineOrchestrator = AnalysisOrchestrator  # Alias for backward compatibility


async def analyze_audio(
    audio: bytes,
    filename: str = "audio.wav",
    content_type: str = "audio/wav",
    request_id: str | None = None,
) -> AnalysisResponse:
    """Convenience functional interface."""
    return await orchestrator.analyze_audio(
        audio, filename=filename, content_type=content_type, request_id=request_id
    )
