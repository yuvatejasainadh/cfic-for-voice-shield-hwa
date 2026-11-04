"""Pydantic schemas governing Cloud Inference inter-service contracts."""

from typing import Optional
from pydantic import BaseModel, Field


class LIDResponse(BaseModel):
    """Normalized response from Language Identification model."""
    status: str = Field(default="success")
    language: str = Field(..., description="ISO 639-1 language code (e.g., 'en', 'te', 'hi')")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    model_version: str = Field(default="lid-scaffold-v0.1")
    latency_ms: float = Field(default=0.0)


class ASRResponse(BaseModel):
    """Normalized response from IndicConformer ASR service."""
    status: str = Field(default="success")
    text: Optional[str] = Field(default=None, description="Decoded transcription text")
    language: str = Field(..., description="Language utilized for ASR decoding")
    model_version: str = Field(default="indicconformer-v0.1")
    latency_ms: float = Field(default=0.0)


class SpoofResponse(BaseModel):
    """Normalized response from VMC-South or AASIST-L acoustic spoof detector."""
    status: str = Field(default="success")
    prediction: Optional[str] = Field(default="GENUINE", description="GENUINE or FAKE")
    score: Optional[float] = Field(default=0.0, description="Synthetic likelihood score")
    model_version: str = Field(default="vmc-south-v0.1")
    latency_ms: float = Field(default=0.0)


class InferencePipelineResponse(BaseModel):
    """Unified response from Cloud Inference Orchestrator to API Gateway."""
    status: str = Field(default="success")
    lid: LIDResponse
    transcription: ASRResponse
    spoof: SpoofResponse
    latency_ms: float = Field(default=0.0)
