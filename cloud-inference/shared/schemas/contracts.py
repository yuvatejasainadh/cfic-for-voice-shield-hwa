"""Pydantic schemas governing Cloud Inference inter-service contracts."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    language: str = Field(..., description="Canonical ISO 639-1 code")
    raw_label: str = Field(..., description="Original model label (e.g. en-IN)")
    confidence: float = Field(..., ge=0.0, le=1.0)


class LIDResponse(BaseModel):
    """Normalized response from Language Identification model (onecxi/open-vakgyata)."""
    status: str = Field(default="success", description="'success', 'silence', 'inconclusive', or 'error'")
    language: str = Field(..., description="ISO 639-1 language code (e.g., 'en', 'te', 'hi')")
    raw_label: Optional[str] = Field(default=None, description="Original model class label (e.g., 'en-IN')")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    probabilities: Optional[Dict[str, float]] = Field(default=None)
    top_k: Optional[List[TopPrediction]] = Field(default=None)
    model_id: str = Field(default="onecxi/open-vakgyata")
    model_revision: Optional[str] = Field(default=None)
    latency_ms: float = Field(default=0.0)
    audio_duration_seconds: Optional[float] = Field(default=0.0)
    device: Optional[str] = Field(default="cpu")


class ASRResponse(BaseModel):
    """Normalized response from ASR service (openai/whisper-base.en or IndicConformer)."""
    status: str = Field(default="success", description="'success', 'silence', or 'error'")
    text: Optional[str] = Field(default=None, description="Decoded transcription text")
    language: str = Field(default="en", description="Target language utilized for ASR decoding")
    model_id: str = Field(default="openai/whisper-base.en")
    model_revision: Optional[str] = Field(default=None)
    latency_ms: float = Field(default=0.0)
    audio_duration_seconds: Optional[float] = Field(default=0.0)
    device: Optional[str] = Field(default="cpu")


class SpoofResponse(BaseModel):
    """Normalized response from acoustic spoof detector."""
    status: str = Field(default="success")
    prediction: Optional[str] = Field(default="GENUINE", description="GENUINE or FAKE")
    score: Optional[float] = Field(default=0.0, description="Synthetic likelihood score")
    model_version: str = Field(default="vmc-south-v0.1")
    latency_ms: float = Field(default=0.0)


class RIASResponse(BaseModel):
    """Normalized response from Gemma RIAS LLM inference service."""
    status: str = Field(default="success", description="'success', 'skipped', 'inconclusive', or 'error'")
    risk_level: str = Field(default="INCONCLUSIVE", description="'LOW', 'MEDIUM', 'HIGH', or 'INCONCLUSIVE'")
    failure_reason: Optional[str] = Field(default=None)
    model_id: str = Field(default="google/gemma-4-E2B-it")
    model_revision: Optional[str] = Field(default="4adf15748de9e1a0097b6274cee9c15836c41c37")
    latency_ms: float = Field(default=0.0)


class InferencePipelineResponse(BaseModel):
    """Unified response from Cloud Inference Orchestrator to API Gateway."""
    status: str = Field(default="success", description="'success', 'inconclusive', or 'error'")
    session_id: Optional[str] = Field(default=None)
    chunk_index: Optional[int] = Field(default=0)
    timestamp_start_s: Optional[float] = Field(default=0.0)
    timestamp_end_s: Optional[float] = Field(default=10.0)
    lid: LIDResponse
    transcription: ASRResponse
    spoof: SpoofResponse
    rias: Optional[RIASResponse] = Field(default=None)
    latency_ms: float = Field(default=0.0)
