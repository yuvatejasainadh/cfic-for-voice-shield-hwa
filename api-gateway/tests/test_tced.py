"""Comprehensive TCED decision policy tests covering Rules A through H."""

from app.schemas import FinalOutcome, RiskLevel
from app.tced.policy import decide
from app.orchestration.decision_engine import DecisionEngine


def test_rule_a_acoustic_fake_rias_low():
    """Rule A: Acoustic FAKE + RIAS LOW -> FAKE (ACOUSTIC_SPOOF_DETECTED)."""
    outcome, reason = decide(
        {"status": "success", "prediction": "FAKE", "score": 0.88},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome == FinalOutcome.FAKE
    assert reason == "ACOUSTIC_SPOOF_DETECTED"


def test_rule_b_acoustic_genuine_rias_low():
    """Rule B: Acoustic GENUINE + RIAS LOW -> GENUINE (ACOUSTIC_GENUINE_CONFIRMED)."""
    outcome, reason = decide(
        {"status": "success", "prediction": "GENUINE", "score": 0.05},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome == FinalOutcome.GENUINE
    assert reason == "ACOUSTIC_GENUINE_CONFIRMED"


def test_rule_c_acoustic_genuine_rias_high():
    """Rule C: Acoustic GENUINE + RIAS HIGH -> SUSPICIOUS (HIGH_RIAS_OVERRIDE)."""
    outcome, reason = decide(
        {"status": "success", "prediction": "GENUINE", "score": 0.05},
        {"status": "success", "risk_level": "HIGH"},
    )
    assert outcome == FinalOutcome.SUSPICIOUS
    assert reason == "HIGH_RIAS_OVERRIDE"


def test_rule_d_acoustic_fake_rias_high():
    """Rule D: Acoustic FAKE + RIAS HIGH -> FAKE (ACOUSTIC_SPOOF_DETECTED)."""
    outcome, reason = decide(
        {"status": "success", "prediction": "FAKE", "score": 0.95},
        {"status": "success", "risk_level": "HIGH"},
    )
    assert outcome == FinalOutcome.FAKE
    assert reason == "ACOUSTIC_SPOOF_DETECTED"


def test_rule_e_acoustic_unavailable_rias_high():
    """Rule E: Acoustic UNAVAILABLE + RIAS HIGH -> SUSPICIOUS (HIGH_RIAS_OVERRIDE)."""
    outcome, reason = decide(
        {"status": "error", "prediction": None},
        {"status": "success", "risk_level": "HIGH"},
    )
    assert outcome == FinalOutcome.SUSPICIOUS
    assert reason == "HIGH_RIAS_OVERRIDE"


def test_rule_f_acoustic_unavailable_rias_low():
    """Rule F: Acoustic UNAVAILABLE + RIAS LOW -> INCONCLUSIVE (ACOUSTIC_EVIDENCE_UNAVAILABLE)."""
    outcome, reason = decide(
        {"status": "error", "prediction": None},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert reason == "ACOUSTIC_EVIDENCE_UNAVAILABLE"


def test_rule_g_both_unavailable():
    """Rule G: Both evidence sources unavailable -> INCONCLUSIVE (INSUFFICIENT_EVIDENCE)."""
    outcome, reason = decide(
        {"status": "error"},
        {"status": "error"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert reason == "INSUFFICIENT_EVIDENCE"


def test_rule_h_invalid_rias_risk_level():
    """Rule H: Unknown or unparseable risk strings -> INCONCLUSIVE (INVALID_RIAS_RISK_LEVEL)."""
    outcome, reason = decide(
        {"status": "success", "prediction": "GENUINE"},
        {"status": "success", "risk_level": "UNKNOWN_RISK_STRING"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert reason == "INVALID_RIAS_RISK_LEVEL"


def test_unvalidated_fusion_is_inconclusive():
    """Before empirical calibration, uncalibrated engine returns TCED_CALIBRATION_TBD."""
    uncalibrated = DecisionEngine(calibrated=False)
    outcome, reason = uncalibrated.decide(
        {"status": "success", "prediction": "GENUINE"},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert reason == "TCED_CALIBRATION_TBD"


def test_missing_evidence_never_silently_genuine():
    """Partial evidence must NEVER silently classify as GENUINE."""
    # Only Pella succeeded with GENUINE
    outcome1, reason1 = decide(
        {"status": "success", "prediction": "GENUINE"},
        {"status": "error", "risk_level": None},
    )
    assert outcome1 == FinalOutcome.INCONCLUSIVE
    assert outcome1 != FinalOutcome.GENUINE
    assert reason1 == "ACOUSTIC_EVIDENCE_UNAVAILABLE"

    # Only RIAS succeeded with LOW
    outcome2, reason2 = decide(
        {"status": "error", "prediction": None},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome2 == FinalOutcome.INCONCLUSIVE
    assert outcome2 != FinalOutcome.GENUINE
    assert reason2 == "ACOUSTIC_EVIDENCE_UNAVAILABLE"
