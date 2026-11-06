"""Pydantic schemas for Gemma RIAS Microservice."""

from typing import Optional
from pydantic import BaseModel, Field


class AssessRequest(BaseModel):
    job_id: Optional[str] = Field(default=None, description="Job or caller session identifier")
    session_id: Optional[str] = Field(default=None, description="Alternative session identifier")
    chunk_index: int = Field(default=0, ge=0, description="Sequential audio chunk index")
    language: Optional[str] = Field(default="en", description="Detected language code (e.g. 'en', 'te', 'hi')")
    transcript: str = Field(..., description="Validated ASR transcript text")

    def get_job_id(self) -> str:
        return self.job_id or self.session_id or "unknown-job"


class AssessResponse(BaseModel):
    job_id: Optional[str] = Field(default=None)
    chunk_index: int = Field(default=0)
    status: str = Field(..., description="'SUCCESS', 'SKIPPED', 'INCONCLUSIVE', or 'ERROR'")
    risk_level: str = Field(..., description="'LOW', 'MEDIUM', 'HIGH', or 'INCONCLUSIVE'")
    failure_reason: Optional[str] = Field(default=None, description="Explains why risk was inconclusive")
    model_id: str = Field(default="google/gemma-4-E2B-it")
    model_revision: Optional[str] = Field(default="4adf15748de9e1a0097b6274cee9c15836c41c37")
    latency_ms: float = Field(default=0.0)


class HealthResponse(BaseModel):
    status: str
    service: str = "model-rias-gemma"
    loaded: bool
    model_id: str
    model_revision: str
    device: str
    is_mock: bool
    error: Optional[str] = None


class ReadinessResponse(BaseModel):
    status: str = "ready"
    loaded: bool = True
    model_id: str
    active_requests: int = 0
    max_concurrency: int = 2


class MetadataResponse(BaseModel):
    service: str = "model-rias-gemma"
    model_id: str = "google/gemma-4-E2B-it"
    model_revision: str = "4adf15748de9e1a0097b6274cee9c15836c41c37"
    quantization: str = "Q4_K_M"
    architecture: str = "Gemma4ForConditionalGeneration"
    runtime: str = "llama.cpp"
    execution: str = "CPU-first"
    n_ctx: int = 2048
    n_threads: int = 4
    allowed_labels: list[str] = ["LOW", "MEDIUM", "HIGH"]
