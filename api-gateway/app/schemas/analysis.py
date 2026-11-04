"""Pydantic schemas for analysis requests and responses."""

from enum import Enum
from pydantic import BaseModel, Field


class FinalOutcome(str, Enum):
    """Normalized final decision from Decision Engine."""

    GENUINE = "GENUINE"
    FAKE = "FAKE"
    SUSPICIOUS = "SUSPICIOUS"
    INCONCLUSIVE = "INCONCLUSIVE"


class RiskLevel(str, Enum):
    """Categorical risk assessment from conversational/semantic risk analysis."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ComponentStatus(str, Enum):
    """Status of downstream analysis components."""

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    NOT_CONFIGURED = "not_configured"


class AnalysisResponse(BaseModel):
    """Normalized analysis response from CFIC API Gateway."""

    request_id: str = Field(..., description="Unique request identifier for tracing")
    final_outcome: FinalOutcome = Field(
        ..., description="Final combined outcome from decision engine"
    )
    reason_code: str = Field(..., description="Machine-readable decision rationale")
    pella_status: str = Field(..., description="Status of acoustic deepfake detector")
    rias_transcription_status: str = Field(
        ..., description="Status of transcription stage"
    )
    rias_risk_status: str = Field(
        ..., description="Status of semantic risk assessment"
    )
    rias_risk_level: RiskLevel | None = Field(
        default=None, description="Assessed semantic risk level (LOW/MEDIUM/HIGH)"
    )
    model_version: str | None = Field(
        default=None, description="Detector / pipeline model version if available"
    )
    latency_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Latency breakdown in milliseconds (total, pella, transcription, rias_llm)",
    )
