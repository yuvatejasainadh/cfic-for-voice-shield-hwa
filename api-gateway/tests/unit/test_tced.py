"""Unit tests for TCED Decision Policy."""

import pytest
from app.schemas import FinalOutcome, ReasonCode, RiskLevel
from app.tced.policy import decide


class TestTCEDPolicy:
    """Comprehensive test suite for TCED decision logic and safety overrides."""

    def test_mandatory_high_rias_override_genuine(self):
        """RIAS HIGH must force SUSPICIOUS even when Pella reports genuine."""
        pella = {"status": "success", "prediction": "GENUINE", "score": 0.02}
        rias = {"status": "success", "risk_level": "HIGH"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.SUSPICIOUS
        assert reason == ReasonCode.HIGH_RIAS_OVERRIDE.value

    def test_mandatory_high_rias_override_fake(self):
        """Rule D: Acoustic FAKE + RIAS HIGH -> FAKE (ACOUSTIC_SPOOF_DETECTED)."""
        pella = {"status": "success", "prediction": "FAKE", "score": 0.98}
        rias = {"status": "success", "risk_level": "HIGH"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.FAKE
        assert reason == ReasonCode.ACOUSTIC_SPOOF_DETECTED.value

    def test_mandatory_high_rias_override_on_detector_failure(self):
        """RIAS HIGH forces SUSPICIOUS when Pella failed or timed out."""
        pella = {"status": "timeout", "prediction": None}
        rias = {"status": "success", "risk_level": "HIGH"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.SUSPICIOUS
        assert reason == ReasonCode.HIGH_RIAS_OVERRIDE.value

    def test_both_paths_failed_returns_inconclusive(self):
        """When both analysis paths fail, TCED returns INCONCLUSIVE."""
        pella = {"status": "error", "prediction": None}
        rias = {"status": "error", "risk_level": None}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == ReasonCode.INSUFFICIENT_EVIDENCE.value

    def test_both_paths_empty_dicts_returns_inconclusive(self):
        """Empty dictionaries should safely yield INSUFFICIENT_EVIDENCE."""
        outcome, reason = decide({}, {})
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == ReasonCode.INSUFFICIENT_EVIDENCE.value

    def test_none_inputs_returns_inconclusive(self):
        """None arguments should safely yield INSUFFICIENT_EVIDENCE."""
        outcome, reason = decide(None, None)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == ReasonCode.INSUFFICIENT_EVIDENCE.value

    def test_pella_success_rias_error_returns_inconclusive(self):
        """Row 9: Acoustic GENUINE + RIAS UNAVAILABLE -> INCONCLUSIVE (RIAS_EVIDENCE_UNAVAILABLE)."""
        pella = {"status": "success", "prediction": "GENUINE", "score": 0.05}
        rias = {"status": "error", "risk_level": None}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == ReasonCode.RIAS_EVIDENCE_UNAVAILABLE.value

    def test_pella_error_rias_low_returns_inconclusive(self):
        """Row 11: Acoustic detector UNAVAILABLE + RIAS LOW -> INCONCLUSIVE (ACOUSTIC_EVIDENCE_UNAVAILABLE)."""
        pella = {"status": "error", "prediction": None}
        rias = {"status": "success", "risk_level": "LOW"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == ReasonCode.ACOUSTIC_EVIDENCE_UNAVAILABLE.value

    def test_pella_error_rias_medium_returns_inconclusive(self):
        """Row 12: Acoustic detector UNAVAILABLE + RIAS MEDIUM -> INCONCLUSIVE (INSUFFICIENT_EVIDENCE)."""
        pella = {"status": "error", "prediction": None}
        rias = {"status": "success", "risk_level": "MEDIUM"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == ReasonCode.INSUFFICIENT_EVIDENCE.value

    def test_genuine_medium_returns_evidence_conflict(self):
        """Row 7: Acoustic GENUINE + RIAS MEDIUM -> INCONCLUSIVE (EVIDENCE_CONFLICT). Never GENUINE."""
        pella = {"status": "success", "prediction": "GENUINE", "score": 0.05}
        rias = {"status": "success", "risk_level": "MEDIUM"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert outcome != FinalOutcome.GENUINE
        assert reason == ReasonCode.EVIDENCE_CONFLICT.value

    def test_fake_invalid_returns_fake(self):
        """Row 5: Acoustic FAKE + RIAS INVALID -> FAKE (ACOUSTIC_SPOOF_DETECTED). Priority 1 absolute."""
        pella = {"status": "success", "prediction": "FAKE", "score": 0.95}
        rias = {"status": "success", "risk_level": "INVALID_RISK"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.FAKE
        assert reason == ReasonCode.ACOUSTIC_SPOOF_DETECTED.value

    def test_invalid_rias_risk_level_is_inconclusive(self):
        """Row 10: Unknown or unparseable risk strings must NEVER default to LOW."""
        pella = {"status": "success", "prediction": "GENUINE", "score": 0.10}
        rias = {"status": "success", "risk_level": "CRITICAL_FRAUD"}
        outcome, reason = decide(pella, rias)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == ReasonCode.INVALID_RIAS_RISK_LEVEL.value

    def test_uncalibrated_fusion_is_inconclusive(self):
        """Dual success before empirical calibration returns TCED_CALIBRATION_TBD."""
        from app.orchestration.decision_engine import DecisionEngine
        uncalibrated = DecisionEngine(calibrated=False)
        pella = {"status": "success", "prediction": "GENUINE", "score": 0.10}
        rias = {"status": "success", "risk_level": "LOW"}
        outcome, reason = uncalibrated.decide(pella, rias)
        assert outcome == FinalOutcome.INCONCLUSIVE
        assert reason == ReasonCode.TCED_CALIBRATION_TBD.value
