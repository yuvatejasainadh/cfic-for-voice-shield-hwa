"""Unit tests for Pydantic schemas and serialization."""

import pytest
from pydantic import ValidationError
from app.schemas import (
    AnalysisResponse,
    ComponentStatus,
    FinalOutcome,
    HealthResponse,
    RiskLevel,
)


class TestSchemas:
    def test_analysis_response_valid(self):
        resp = AnalysisResponse(
            request_id="test-req-123",
            final_outcome=FinalOutcome.SUSPICIOUS,
            reason_code="HIGH_RIAS_OVERRIDE",
            pella_status=ComponentStatus.SUCCESS.value,
            rias_transcription_status=ComponentStatus.SUCCESS.value,
            rias_risk_status=ComponentStatus.SUCCESS.value,
            rias_risk_level=RiskLevel.HIGH,
            model_version="pellav2-v1",
            latency_ms={"total": 250.0, "pella": 100.0},
        )
        data = resp.model_dump(mode="json")
        assert data["request_id"] == "test-req-123"
        assert data["final_outcome"] == "SUSPICIOUS"
        assert data["rias_risk_level"] == "HIGH"
        assert data["latency_ms"]["total"] == 250.0

    def test_analysis_response_invalid_outcome(self):
        with pytest.raises(ValidationError):
            AnalysisResponse(
                request_id="test-req-123",
                final_outcome="INVALID_OUTCOME",  # Not in FinalOutcome enum
                reason_code="test",
                pella_status="error",
                rias_transcription_status="error",
                rias_risk_status="error",
            )

    def test_health_response_valid(self):
        health = HealthResponse(
            status="ok",
            service="cfic-api",
            version="0.1.0",
            environment="testing",
            downstream_status={"pella": "ok"},
        )
        assert health.status == "ok"
        assert health.environment == "testing"
