"""Original root TCED tests, updated and expanded."""

from app.schemas import FinalOutcome, RiskLevel
from app.tced.policy import decide


def test_high_rias_always_suspicious():
    """Mandatory Locked Rule: RIAS HIGH always yields SUSPICIOUS."""
    outcome, reason = decide(
        {"status": "success", "prediction": "GENUINE", "score": 0.05},
        {"status": "success", "risk_level": "HIGH"},
    )
    assert outcome == FinalOutcome.SUSPICIOUS
    assert reason == "HIGH_RIAS_OVERRIDE"


def test_high_rias_overrides_fake():
    """RIAS HIGH overrides even when Pella predicts FAKE."""
    outcome, reason = decide(
        {"status": "success", "prediction": "FAKE", "score": 0.95},
        {"status": "success", "risk_level": "HIGH"},
    )
    assert outcome == FinalOutcome.SUSPICIOUS
    assert reason == "HIGH_RIAS_OVERRIDE"


def test_high_rias_overrides_pella_error():
    """RIAS HIGH overrides even when Pella detector fails or times out."""
    outcome, reason = decide(
        {"status": "error", "prediction": None},
        {"status": "success", "risk_level": "HIGH"},
    )
    assert outcome == FinalOutcome.SUSPICIOUS
    assert reason == "HIGH_RIAS_OVERRIDE"


def test_both_failed_inconclusive():
    """When both analysis paths fail, TCED returns INCONCLUSIVE."""
    outcome, reason = decide(
        {"status": "error"},
        {"status": "error"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert reason == "INSUFFICIENT_EVIDENCE"


def test_unvalidated_fusion_is_inconclusive():
    """Before empirical calibration, dual success returns INCONCLUSIVE."""
    outcome, reason = decide(
        {"status": "success", "prediction": "native"},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert reason == "TCED_CALIBRATION_TBD"


def test_missing_evidence_never_silently_genuine():
    """Partial evidence must NEVER silently classify as GENUINE."""
    # Only Pella succeeded
    outcome1, reason1 = decide(
        {"status": "success", "prediction": "GENUINE"},
        {"status": "error", "risk_level": None},
    )
    assert outcome1 == FinalOutcome.INCONCLUSIVE
    assert outcome1 != FinalOutcome.GENUINE

    # Only RIAS succeeded with LOW
    outcome2, reason2 = decide(
        {"status": "error", "prediction": None},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome2 == FinalOutcome.INCONCLUSIVE
    assert outcome2 != FinalOutcome.GENUINE


def test_invalid_rias_risk_never_defaults_to_low():
    """Unknown or invalid risk level must NEVER default to LOW."""
    outcome, reason = decide(
        {"status": "success", "prediction": "GENUINE"},
        {"status": "success", "risk_level": "UNKNOWN_RISK_STRING"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert reason == "INVALID_RIAS_RISK_LEVEL"
