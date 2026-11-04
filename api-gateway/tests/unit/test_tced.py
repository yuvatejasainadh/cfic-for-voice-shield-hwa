"""Unit tests for TCED Decision Policy."""

import pytest
from app.schemas import FinalOutcome, RiskLevel
from app.tced.policy import decide


class TestTCEDPolicy:
    """Comprehensive test suite for TCED decision logic and safety overrides."""

    def test_mandatory_high_rias_override_genuine(self):
        """RIAS HIGH must force SUSPICIOUS even when Pella reports genuine."""
        pella = {"status": "success", "prediction": "GENUINE", "score": 0.02}
        rias = {"status": "success", "risk_level": "HIGH"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.SUSPICIOUS
        assert reason == "HIGH_RIAS_OVERRIDE"

    def test_mandatory_high_rias_override_fake(self):
        """RIAS HIGH forces SUSPICIOUS when Pella reports fake."""
        pella = {"status": "success", "prediction": "FAKE", "score": 0.98}
        rias = {"status": "success", "risk_level": "HIGH"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.SUSPICIOUS
        assert reason == "HIGH_RIAS_OVERRIDE"

    def test_mandatory_high_rias_override_on_detector_failure(self):
        """RIAS HIGH forces SUSPICIOUS when Pella failed or timed out."""
        pella = {"status": "timeout", "prediction": None}
        rias = {"status": "success", "risk_level": "HIGH"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.SUSPICIOUS
        assert reason == "HIGH_RIAS_OVERRIDE"

    def test_both_paths_failed_returns_inconclusive(self):
        """When both analysis paths fail, TCED returns INCONCLUSIVE."""
        pella = {"status": "error", "prediction": None}
        rias = {"status": "error", "risk_level": None}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == "INSUFFICIENT_EVIDENCE"

    def test_both_paths_empty_dicts_returns_inconclusive(self):
        """Empty dictionaries should safely yield INSUFFICIENT_EVIDENCE."""
        outcome, reason = decide({}, {})
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == "INSUFFICIENT_EVIDENCE"

    def test_none_inputs_returns_inconclusive(self):
        """None arguments should safely yield INSUFFICIENT_EVIDENCE."""
        outcome, reason = decide(None, None)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == "INSUFFICIENT_EVIDENCE"

    def test_pella_success_rias_error_returns_inconclusive(self):
        """Partial evidence (Pella ok, RIAS failed) must not silently claim genuine."""
        pella = {"status": "success", "prediction": "GENUINE", "score": 0.05}
        rias = {"status": "error", "risk_level": None}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == "PARTIAL_EVIDENCE_POLICY_TBD"

    def test_pella_error_rias_low_returns_inconclusive(self):
        """Partial evidence (Pella failed, RIAS ok) must not silently claim genuine."""
        pella = {"status": "error", "prediction": None}
        rias = {"status": "success", "risk_level": "LOW"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == "PARTIAL_EVIDENCE_POLICY_TBD"

    def test_pella_error_rias_medium_returns_inconclusive(self):
        pella = {"status": "error", "prediction": None}
        rias = {"status": "success", "risk_level": "MEDIUM"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == "PARTIAL_EVIDENCE_POLICY_TBD"

    def test_invalid_rias_risk_level_is_inconclusive(self):
        """Unknown or unparseable risk strings must NEVER default to LOW."""
        pella = {"status": "success", "prediction": "GENUINE", "score": 0.10}
        rias = {"status": "success", "risk_level": "CRITICAL_FRAUD"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == "INVALID_RIAS_RISK_LEVEL"

    def test_uncalibrated_fusion_is_inconclusive(self):
        """Dual success before empirical calibration returns TCED_CALIBRATION_TBD."""
        pella = {"status": "success", "prediction": "GENUINE", "score": 0.10}
        rias = {"status": "success", "risk_level": "LOW"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == "TCED_CALIBRATION_TBD"
