"""Pydantic schemas for PellaV2 Acoustic Deepfake Detection Service."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class SpoofPredictionResponse(BaseModel):
    """Normalized detection response conforming to CFIC acoustic spoof contract."""
    status: str = Field(default="success", description="'success', 'error', or 'silence'")
    prediction: Optional[str] = Field(default=None, description="'GENUINE' or 'FAKE'")
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Calibrated synthetic probability")
    threshold: float = Field(default=0.50, description="Decision threshold (score >= threshold => FAKE)")
    model_version: str = Field(default="pellav2-original", description="Model identifier")
    latency_ms: float = Field(default=0.0, description="Inference latency in milliseconds")


class SpoofHealthResponse(BaseModel):
    """Service health response schema."""
    status: str = Field(default="ok")
    service: str = Field(default="model-pellav2")
    architecture: str = Field(default="PellaV2-w2v-bert-2.0")
    target_domain: str = Field(default="Acoustic-Synthetic-Detection")
    model_version: str = Field(default="pellav2-original")
    artifact_found: bool = Field(default=False)
    checksum_verified: bool = Field(default=False)
    is_mock: bool = Field(default=False)


class ReadinessResponse(BaseModel):
    """Container readiness response schema."""
    status: str = Field(default="ready")
    ready: bool = Field(default=True)
    model_loaded: bool = Field(default=False)


class SpoofMetadataResponse(BaseModel):
    """Detailed model metadata and configuration."""
    service: str = "model-pellav2"
    model_id: str = "Sadanie/pellav2-audio-deepfake-detector"
    model_version: str = "pellav2-original"
    architecture: str = "PellaV2-w2v-bert-2.0"
    quantization: str = "None (FP32)"
    sample_rate: int = 16000
    feature_dim: int = 160
    threshold: float = 0.50
    calibration_parameters: Dict[str, float] = Field(default_factory=dict)
