"""Pydantic schemas for Cloud Inference Orchestrator."""

from typing import Optional
from pydantic import BaseModel, Field


class LIDResult(BaseModel):
    status: str = Field(default="success")
    language: str = Field(..., description="Detected language code")
    confidence: float = Field(default=1.0)
    model_version: str = Field(default="lid-v0.1")
    latency_ms: float = Field(default=0.0)


class ASRResult(BaseModel):
    status: str = Field(default="success")
    text: Optional[str] = Field(default=None)
    language: str = Field(...)
    model_version: str = Field(default="indicconformer-v0.1")
    latency_ms: float = Field(default=0.0)


class SpoofResult(BaseModel):
    status: str = Field(default="success")
    prediction: Optional[str] = Field(default="GENUINE")
    score: Optional[float] = Field(default=0.0)
    model_version: str = Field(default="spoof-v0.1")
    latency_ms: float = Field(default=0.0)


class InferenceResponse(BaseModel):
    status: str = Field(default="success")
    lid: LIDResult
    transcription: ASRResult
    spoof: SpoofResult
    latency_ms: float = Field(default=0.0)


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    service: str = Field(default="cloud-inference-orchestrator")
    version: str = Field(default="0.1.0")
    backends: dict[str, str] = Field(default_factory=dict)
