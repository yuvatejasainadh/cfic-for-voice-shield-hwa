"""Pydantic schemas for Cloud Inference Orchestrator."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TopCandidate(BaseModel):
    language: str
    raw_label: str
    confidence: float


class LIDResult(BaseModel):
    status: str = Field(default="success", description="'success', 'silence', 'inconclusive', 'error'")
    language: str = Field(..., description="Detected canonical language code")
    raw_label: Optional[str] = Field(default=None, description="Original model label (e.g. en-IN)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    probabilities: Optional[Dict[str, float]] = Field(default=None)
    top_k: Optional[List[Dict[str, Any]]] = Field(default=None)
    model_id: Optional[str] = Field(default="onecxi/open-vakgyata")
    model_revision: Optional[str] = Field(default=None)
    retries_used: int = Field(default=0, description="Number of LID retry attempts used")
    latency_ms: float = Field(default=0.0)


class ASRResult(BaseModel):
    status: str = Field(default="success", description="'success', 'skipped', or 'error'")
    text: Optional[str] = Field(default=None, description="Decoded transcript text")
    language: str = Field(default="en", description="Decoded language")
    model_id: Optional[str] = Field(default="openai/whisper-base.en")
    model_revision: Optional[str] = Field(default=None)
    latency_ms: float = Field(default=0.0)


class SpoofResult(BaseModel):
    status: str = Field(default="success")
    prediction: Optional[str] = Field(default="GENUINE")
    score: Optional[float] = Field(default=0.0)
    model_version: str = Field(default="spoof-v0.1")
    latency_ms: float = Field(default=0.0)


class RIASResult(BaseModel):
    status: str = Field(default="success", description="'success', 'skipped', 'inconclusive', or 'error'")
    risk_level: str = Field(default="INCONCLUSIVE", description="'LOW', 'MEDIUM', 'HIGH', or 'INCONCLUSIVE'")
    failure_reason: Optional[str] = Field(default=None, description="Reason if risk is inconclusive or skipped")
    model_id: Optional[str] = Field(default="google/gemma-4-E2B-it")
    model_revision: Optional[str] = Field(default="4adf15748de9e1a0097b6274cee9c15836c41c37")
    latency_ms: float = Field(default=0.0)


class InferenceResponse(BaseModel):
    status: str = Field(default="success", description="'success', 'inconclusive', or 'error'")
    session_id: Optional[str] = Field(default=None, description="Client session identifier")
    chunk_index: Optional[int] = Field(default=0, description="Sequential index of audio chunk")
    timestamp_start_s: Optional[float] = Field(default=0.0, description="Chunk start timestamp in seconds")
    timestamp_end_s: Optional[float] = Field(default=10.0, description="Chunk end timestamp in seconds")
    lid: LIDResult
    transcription: ASRResult
    spoof: SpoofResult
    rias: Optional[RIASResult] = Field(default=None, description="RIAS conversational semantic risk assessment")
    latency_ms: float = Field(default=0.0)


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    service: str = "cloud-inference-orchestrator"
    version: str = "0.2.0"
    backends: Dict[str, str] = Field(default_factory=dict)
