"""Analysis Orchestrator coordinating audio ingestion, inference services, and decision fusion."""

import asyncio
import logging
import time
import uuid
from typing import Any, List
from app.core.config import settings
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

logger = logging.getLogger("analysis_orchestrator")


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

        # 1. 5-Second Chunking Orchestration (Phase 4)
        chunks = audio_service.slice_into_5s_chunks(
            audio_bytes=audio,
            filename=filename,
            content_type=content_type,
            chunk_duration_s=5.0,
        )

        # 2. Attempt Cloud Inference Subsystem if configured
        use_cloud = getattr(settings, "use_cloud_inference", True) and bool(self.cloud_client.base_url)

        if use_cloud:
            try:
                chunk_responses: List[dict[str, Any]] = []
                for chunk in chunks:
                    c_resp = await self.cloud_client.run_inference(
                        audio=chunk["audio_bytes"],
                        filename=chunk["filename"],
                        content_type=chunk["content_type"],
                        request_id=req_id,
                        chunk_index=chunk["chunk_index"],
                        timestamp_start_s=chunk["start_seconds"],
                        timestamp_end_s=chunk["end_seconds"],
                    )
                    chunk_responses.append(c_resp)

                # If cloud client was reachable and responded with valid inference
                first_status = chunk_responses[0].get("status") if chunk_responses else "error"
                if first_status in ("success", "inconclusive"):
                    return self._aggregate_cloud_responses(
                        req_id=req_id,
                        chunk_responses=chunk_responses,
                        pipeline_start=pipeline_start,
                    )
                else:
                    logger.warning(
                        f"Cloud inference returned {first_status}; falling back to local/legacy pipeline."
                    )
            except Exception as exc:
                logger.warning(f"Cloud inference dispatch failed ({exc}); falling back to legacy pipeline.")

        # 3. Fallback: Legacy Parallel Dispatch
        return await self._run_legacy_pipeline(
            audio=audio,
            filename=filename,
            content_type=content_type,
            req_id=req_id,
            pipeline_start=pipeline_start,
        )

    def _aggregate_cloud_responses(
        self,
        req_id: str,
        chunk_responses: List[dict[str, Any]],
        pipeline_start: float,
    ) -> AnalysisResponse:
        """Aggregate chunk-level Cloud Inference responses into final session verdict."""
        latencies: dict[str, float] = {}
        ordered_chunks = sorted(chunk_responses, key=lambda c: c.get("chunk_index", 0))

        # 1. Aggregate transcriptions in sequential chunk order
        transcription_texts: List[str] = []
        last_model_id = None
        transcription_status = "success"
        transcription_latency = 0.0

        for c in ordered_chunks:
            t_obj = c.get("transcription", {})
            if t_obj.get("status") == "error":
                transcription_status = "error"
            if t_obj.get("text"):
                transcription_texts.append(t_obj["text"].strip())
            if t_obj.get("model_id"):
                last_model_id = t_obj["model_id"]
            transcription_latency += t_obj.get("latency_ms", 0.0)

        aggregated_text = " ".join(transcription_texts).strip() or None
        latencies["transcription"] = round(transcription_latency, 2)

        transc_res = {
            "status": transcription_status if aggregated_text or transcription_status == "success" else "error",
            "text": aggregated_text,
            "language": ordered_chunks[0].get("transcription", {}).get("language", "en"),
            "model_version": last_model_id,
            "latency_ms": latencies["transcription"],
        }

        # 2. Aggregate Spoof verdicts
        spoof_predictions = [c.get("spoof", {}).get("prediction") for c in ordered_chunks if c.get("spoof", {}).get("prediction")]
        spoof_scores = [c.get("spoof", {}).get("score") for c in ordered_chunks if c.get("spoof", {}).get("score") is not None]
        spoof_statuses = [c.get("spoof", {}).get("status") for c in ordered_chunks]
        spoof_latency = sum(c.get("spoof", {}).get("latency_ms", 0.0) for c in ordered_chunks)
        latencies["pella"] = round(spoof_latency, 2)

        if "FAKE" in spoof_predictions:
            final_spoof_pred = "FAKE"
            final_spoof_score = max(spoof_scores) if spoof_scores else 0.95
        elif all(p == "GENUINE" for p in spoof_predictions) and spoof_predictions:
            final_spoof_pred = "GENUINE"
            final_spoof_score = min(spoof_scores) if spoof_scores else 0.05
        else:
            final_spoof_pred = None
            final_spoof_score = None

        final_spoof_status = "success" if final_spoof_pred else ("error" if "error" in spoof_statuses else "inconclusive")
        pella_res = {
            "status": final_spoof_status,
            "prediction": final_spoof_pred,
            "score": final_spoof_score,
            "model_version": ordered_chunks[0].get("spoof", {}).get("model_version", "vmc-south"),
            "latency_ms": latencies["pella"],
        }

        # 3. Aggregate RIAS semantic risk verdicts
        rias_risks = [c.get("rias", {}).get("risk_level") for c in ordered_chunks if c.get("rias", {}).get("risk_level")]
        rias_statuses = [c.get("rias", {}).get("status") for c in ordered_chunks]
        rias_latency = sum(c.get("rias", {}).get("latency_ms", 0.0) for c in ordered_chunks)
        latencies["rias_llm"] = round(rias_latency, 2)

        # TCED Mandatory Rule: Any HIGH risk forces overall HIGH
        if "HIGH" in rias_risks:
            final_risk = "HIGH"
            final_rias_status = "success"
        elif "MEDIUM" in rias_risks:
            final_risk = "MEDIUM"
            final_rias_status = "success"
        elif all(r == "LOW" for r in rias_risks) and rias_risks:
            final_risk = "LOW"
            final_rias_status = "success"
        else:
            final_risk = None
            final_rias_status = "skipped" if all(s == "skipped" for s in rias_statuses) else "inconclusive"

        rias_res = {
            "status": final_rias_status,
            "risk_level": final_risk,
            "reason": ordered_chunks[0].get("rias", {}).get("failure_reason"),
            "latency_ms": latencies["rias_llm"],
        }

        # 4. Synthesize final decision via DecisionEngine
        outcome, reason = decide(pella_res, rias_res)

        total_latency = round((time.perf_counter() - pipeline_start) * 1000, 2)
        latencies["total"] = total_latency

        risk_level_enum = None
        if final_risk:
            try:
                risk_level_enum = RiskLevel(final_risk)
            except ValueError:
                risk_level_enum = None

        response = AnalysisResponse(
            request_id=req_id,
            final_outcome=outcome,
            reason_code=reason,
            pella_status=pella_res["status"],
            rias_transcription_status=transc_res["status"],
            rias_risk_status=rias_res["status"],
            rias_risk_level=risk_level_enum,
            model_version=pella_res.get("model_version"),
            latency_ms=latencies,
        )

        job_service.complete_job(req_id, response.model_dump())
        return response

    async def _run_legacy_pipeline(
        self,
        audio: bytes,
        filename: str,
        content_type: str,
        req_id: str,
        pipeline_start: float,
    ) -> AnalysisResponse:
        """Run legacy parallel dispatch for backward-compatible testing."""
        latencies: dict[str, float] = {}

        # Parallel dispatch of acoustic detector and transcription
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

        # Semantic Risk Assessment via RiskService (triggered if transcription produced text)
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

        # Decision Engine Synthesis
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
