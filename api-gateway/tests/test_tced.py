"""Comprehensive TCED decision policy tests covering all 15 matrix combinations,

rule precedence, edge cases, schema stability, and parity between production
and legacy engines.
"""

import os
import sys

# Ensure repository root is in sys.path for legacy policy import
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest
from app.schemas import FinalOutcome, ReasonCode, RiskLevel
from app.tced.policy import decide
from app.orchestration.decision_engine import DecisionEngine
import legacy.tced.policy as legacy_policy


# ==============================================================================
# SECTION 1: THE 15 AUTHORITATIVE MATRIX COMBINATION TESTS
# ==============================================================================

# --- Acoustic FAKE tests (Rows 1-5) ---

def test_combination_01_fake_low():
    """Row 1: FAKE + LOW -> FAKE (ACOUSTIC_SPOOF_DETECTED)."""
    outcome, reason = decide(
        {"status": "success", "prediction": "FAKE", "score": 0.92},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome == FinalOutcome.FAKE
    assert reason == ReasonCode.ACOUSTIC_SPOOF_DETECTED.value


def test_combination_02_fake_medium():
    """Row 2: FAKE + MEDIUM -> FAKE (ACOUSTIC_SPOOF_DETECTED)."""
    outcome, reason = decide(
        {"status": "success", "prediction": "FAKE", "score": 0.88},
        {"status": "success", "risk_level": "MEDIUM"},
    )
    assert outcome == FinalOutcome.FAKE
    assert reason == ReasonCode.ACOUSTIC_SPOOF_DETECTED.value


def test_combination_03_fake_high():
    """Row 3: FAKE + HIGH -> FAKE (ACOUSTIC_SPOOF_DETECTED). Priority 1 over RIAS HIGH."""
    outcome, reason = decide(
        {"status": "success", "prediction": "FAKE", "score": 0.99},
        {"status": "success", "risk_level": "HIGH"},
    )
    assert outcome == FinalOutcome.FAKE
    assert reason == ReasonCode.ACOUSTIC_SPOOF_DETECTED.value


def test_combination_04_fake_unavailable():
    """Row 4: FAKE + UNAVAILABLE -> FAKE (ACOUSTIC_SPOOF_DETECTED)."""
    outcome, reason = decide(
        {"status": "success", "prediction": "FAKE", "score": 0.85},
        {"status": "error", "risk_level": None},
    )
    assert outcome == FinalOutcome.FAKE
    assert reason == ReasonCode.ACOUSTIC_SPOOF_DETECTED.value


def test_combination_05_fake_invalid():
    """Row 5: FAKE + INVALID -> FAKE (ACOUSTIC_SPOOF_DETECTED). FAKE priority over invalid RIAS."""
    outcome, reason = decide(
        {"status": "success", "prediction": "FAKE", "score": 0.95},
        {"status": "success", "risk_level": "UNKNOWN_RISK_LEVEL"},
    )
    assert outcome == FinalOutcome.FAKE
    assert reason == ReasonCode.ACOUSTIC_SPOOF_DETECTED.value


# --- Acoustic GENUINE tests (Rows 6-10) ---

def test_combination_06_genuine_low():
    """Row 6: GENUINE + LOW -> GENUINE (ACOUSTIC_GENUINE_CONFIRMED). The ONLY GENUINE path."""
    outcome, reason = decide(
        {"status": "success", "prediction": "GENUINE", "score": 0.02},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome == FinalOutcome.GENUINE
    assert reason == ReasonCode.ACOUSTIC_GENUINE_CONFIRMED.value


def test_combination_07_genuine_medium():
    """Row 7: GENUINE + MEDIUM -> INCONCLUSIVE (EVIDENCE_CONFLICT). MEDIUM must NEVER yield GENUINE."""
    outcome, reason = decide(
        {"status": "success", "prediction": "GENUINE", "score": 0.05},
        {"status": "success", "risk_level": "MEDIUM"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert outcome != FinalOutcome.GENUINE
    assert reason == ReasonCode.EVIDENCE_CONFLICT.value


def test_combination_08_genuine_high():
    """Row 8: GENUINE + HIGH -> SUSPICIOUS (HIGH_RIAS_OVERRIDE). Priority 2 overrides GENUINE."""
    outcome, reason = decide(
        {"status": "success", "prediction": "GENUINE", "score": 0.01},
        {"status": "success", "risk_level": "HIGH"},
    )
    assert outcome == FinalOutcome.SUSPICIOUS
    assert reason == ReasonCode.HIGH_RIAS_OVERRIDE.value


def test_combination_09_genuine_unavailable():
    """Row 9: GENUINE + UNAVAILABLE -> INCONCLUSIVE (RIAS_EVIDENCE_UNAVAILABLE)."""
    outcome, reason = decide(
        {"status": "success", "prediction": "GENUINE", "score": 0.04},
        {"status": "timeout", "risk_level": None},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert outcome != FinalOutcome.GENUINE
    assert reason == ReasonCode.RIAS_EVIDENCE_UNAVAILABLE.value


def test_combination_10_genuine_invalid():
    """Row 10: GENUINE + INVALID -> INCONCLUSIVE (INVALID_RIAS_RISK_LEVEL)."""
    outcome, reason = decide(
        {"status": "success", "prediction": "GENUINE", "score": 0.03},
        {"status": "success", "risk_level": "INVALID_RISK_STRING"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert outcome != FinalOutcome.GENUINE
    assert reason == ReasonCode.INVALID_RIAS_RISK_LEVEL.value


# --- Acoustic UNAVAILABLE tests (Rows 11-15) ---

def test_combination_11_unavailable_low():
    """Row 11: UNAVAILABLE + LOW -> INCONCLUSIVE (ACOUSTIC_EVIDENCE_UNAVAILABLE)."""
    outcome, reason = decide(
        {"status": "error", "prediction": None},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert outcome != FinalOutcome.GENUINE
    assert reason == ReasonCode.ACOUSTIC_EVIDENCE_UNAVAILABLE.value


def test_combination_12_unavailable_medium():
    """Row 12: UNAVAILABLE + MEDIUM -> INCONCLUSIVE (INSUFFICIENT_EVIDENCE)."""
    outcome, reason = decide(
        {"status": "error", "prediction": None},
        {"status": "success", "risk_level": "MEDIUM"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert outcome != FinalOutcome.GENUINE
    assert reason == ReasonCode.INSUFFICIENT_EVIDENCE.value


def test_combination_13_unavailable_high():
    """Row 13: UNAVAILABLE + HIGH -> SUSPICIOUS (HIGH_RIAS_OVERRIDE). Priority 2 overrides."""
    outcome, reason = decide(
        {"status": "timeout", "prediction": None},
        {"status": "success", "risk_level": "HIGH"},
    )
    assert outcome == FinalOutcome.SUSPICIOUS
    assert reason == ReasonCode.HIGH_RIAS_OVERRIDE.value


def test_combination_14_unavailable_unavailable():
    """Row 14: UNAVAILABLE + UNAVAILABLE -> INCONCLUSIVE (INSUFFICIENT_EVIDENCE)."""
    outcome, reason = decide(
        {"status": "error", "prediction": None},
        {"status": "error", "risk_level": None},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert reason == ReasonCode.INSUFFICIENT_EVIDENCE.value


def test_combination_15_unavailable_invalid():
    """Row 15: UNAVAILABLE + INVALID -> INCONCLUSIVE (INVALID_RIAS_RISK_LEVEL)."""
    outcome, reason = decide(
        {"status": "error", "prediction": None},
        {"status": "success", "risk_level": "CORRUPTED_VALUE"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert reason == ReasonCode.INVALID_RIAS_RISK_LEVEL.value


# ==============================================================================
# SECTION 2: TABLE-DRIVEN MATRIX VERIFICATION & PARITY
# ==============================================================================

MATRIX_TEST_CASES = [
    # (Acoustic dict, RIAS dict, expected_outcome, expected_reason)
    ({"status": "success", "prediction": "FAKE"}, {"status": "success", "risk_level": "LOW"}, FinalOutcome.FAKE, "ACOUSTIC_SPOOF_DETECTED"),
    ({"status": "success", "prediction": "FAKE"}, {"status": "success", "risk_level": "MEDIUM"}, FinalOutcome.FAKE, "ACOUSTIC_SPOOF_DETECTED"),
    ({"status": "success", "prediction": "FAKE"}, {"status": "success", "risk_level": "HIGH"}, FinalOutcome.FAKE, "ACOUSTIC_SPOOF_DETECTED"),
    ({"status": "success", "prediction": "FAKE"}, {"status": "error"}, FinalOutcome.FAKE, "ACOUSTIC_SPOOF_DETECTED"),
    ({"status": "success", "prediction": "FAKE"}, {"status": "success", "risk_level": "BAD"}, FinalOutcome.FAKE, "ACOUSTIC_SPOOF_DETECTED"),
    ({"status": "success", "prediction": "GENUINE"}, {"status": "success", "risk_level": "LOW"}, FinalOutcome.GENUINE, "ACOUSTIC_GENUINE_CONFIRMED"),
    ({"status": "success", "prediction": "GENUINE"}, {"status": "success", "risk_level": "MEDIUM"}, FinalOutcome.INCONCLUSIVE, "EVIDENCE_CONFLICT"),
    ({"status": "success", "prediction": "GENUINE"}, {"status": "success", "risk_level": "HIGH"}, FinalOutcome.SUSPICIOUS, "HIGH_RIAS_OVERRIDE"),
    ({"status": "success", "prediction": "GENUINE"}, {"status": "error"}, FinalOutcome.INCONCLUSIVE, "RIAS_EVIDENCE_UNAVAILABLE"),
    ({"status": "success", "prediction": "GENUINE"}, {"status": "success", "risk_level": "BAD"}, FinalOutcome.INCONCLUSIVE, "INVALID_RIAS_RISK_LEVEL"),
    ({"status": "error"}, {"status": "success", "risk_level": "LOW"}, FinalOutcome.INCONCLUSIVE, "ACOUSTIC_EVIDENCE_UNAVAILABLE"),
    ({"status": "error"}, {"status": "success", "risk_level": "MEDIUM"}, FinalOutcome.INCONCLUSIVE, "INSUFFICIENT_EVIDENCE"),
    ({"status": "error"}, {"status": "success", "risk_level": "HIGH"}, FinalOutcome.SUSPICIOUS, "HIGH_RIAS_OVERRIDE"),
    ({"status": "error"}, {"status": "error"}, FinalOutcome.INCONCLUSIVE, "INSUFFICIENT_EVIDENCE"),
    ({"status": "error"}, {"status": "success", "risk_level": "BAD"}, FinalOutcome.INCONCLUSIVE, "INVALID_RIAS_RISK_LEVEL"),
]


@pytest.mark.parametrize("acoustic_in, rias_in, expected_outcome, expected_reason", MATRIX_TEST_CASES)
def test_authoritative_matrix_table_driven(acoustic_in, rias_in, expected_outcome, expected_reason):
    """Verify all 15 authoritative matrix cells in production DecisionEngine."""
    outcome, reason = decide(acoustic_in, rias_in)
    assert outcome == expected_outcome
    assert reason == expected_reason


@pytest.mark.parametrize("acoustic_in, rias_in, expected_outcome, expected_reason", MATRIX_TEST_CASES)
def test_legacy_policy_parity(acoustic_in, rias_in, expected_outcome, expected_reason):
    """Verify exact parity between legacy and production decision implementations."""
    legacy_outcome, legacy_reason = legacy_policy.decide(acoustic_in, rias_in)
    prod_outcome, prod_reason = decide(acoustic_in, rias_in)
    assert legacy_outcome == prod_outcome == expected_outcome
    assert legacy_reason == prod_reason == expected_reason


# ==============================================================================
# SECTION 3: RULE PRECEDENCE, DEFENSE-IN-DEPTH & EDGE CASES
# ==============================================================================

def test_priority1_fake_cannot_be_overridden_by_high_rias():
    """Acoustic FAKE cannot be downgraded or altered by HIGH RIAS."""
    outcome, reason = decide(
        {"status": "success", "prediction": "FAKE", "score": 0.999},
        {"status": "success", "risk_level": "HIGH"},
    )
    assert outcome == FinalOutcome.FAKE
    assert reason == ReasonCode.ACOUSTIC_SPOOF_DETECTED.value


def test_priority2_high_rias_cannot_be_downgraded_by_genuine():
    """HIGH conversational risk forces SUSPICIOUS even if acoustic claims GENUINE."""
    outcome, reason = decide(
        {"status": "success", "prediction": "GENUINE", "score": 0.001},
        {"status": "success", "risk_level": "HIGH"},
    )
    assert outcome == FinalOutcome.SUSPICIOUS
    assert reason == ReasonCode.HIGH_RIAS_OVERRIDE.value


def test_priority4_medium_risk_never_yields_genuine():
    """Acoustic GENUINE + RIAS MEDIUM must return EVIDENCE_CONFLICT, NEVER GENUINE."""
    outcome, reason = decide(
        {"status": "success", "prediction": "GENUINE", "score": 0.005},
        {"status": "success", "risk_level": "MEDIUM"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert outcome != FinalOutcome.GENUINE
    assert reason == ReasonCode.EVIDENCE_CONFLICT.value


def test_none_and_empty_inputs_handled_safely():
    """None and empty dictionary inputs return INSUFFICIENT_EVIDENCE."""
    outcome1, reason1 = decide(None, None)
    assert outcome1 == FinalOutcome.INCONCLUSIVE
    assert reason1 == ReasonCode.INSUFFICIENT_EVIDENCE.value

    outcome2, reason2 = decide({}, {})
    assert outcome2 == FinalOutcome.INCONCLUSIVE
    assert reason2 == ReasonCode.INSUFFICIENT_EVIDENCE.value

    outcome3, reason3 = decide(None, {"status": "error"})
    assert outcome3 == FinalOutcome.INCONCLUSIVE
    assert reason3 == ReasonCode.INSUFFICIENT_EVIDENCE.value


def test_case_insensitivity_and_whitespace():
    """Input string variations (case, whitespace) are normalized properly."""
    outcome, reason = decide(
        {"status": "SUCCESS", "prediction": " fake "},
        {"status": "Success", "risk_level": " low "},
    )
    assert outcome == FinalOutcome.FAKE
    assert reason == ReasonCode.ACOUSTIC_SPOOF_DETECTED.value

    outcome2, reason2 = decide(
        {"status": "success", "prediction": "genuine"},
        {"status": "success", "risk_level": "high"},
    )
    assert outcome2 == FinalOutcome.SUSPICIOUS
    assert reason2 == ReasonCode.HIGH_RIAS_OVERRIDE.value


def test_risk_level_enum_object_support():
    """RiskLevel enum instances are handled identically to strings."""
    outcome, reason = decide(
        {"status": "success", "prediction": "GENUINE"},
        {"status": "success", "risk_level": RiskLevel.LOW},
    )
    assert outcome == FinalOutcome.GENUINE
    assert reason == ReasonCode.ACOUSTIC_GENUINE_CONFIRMED.value

    outcome_med, reason_med = decide(
        {"status": "success", "prediction": "GENUINE"},
        {"status": "success", "risk_level": RiskLevel.MEDIUM},
    )
    assert outcome_med == FinalOutcome.INCONCLUSIVE
    assert reason_med == ReasonCode.EVIDENCE_CONFLICT.value


def test_uncalibrated_fusion_is_inconclusive():
    """Before empirical calibration, uncalibrated engine returns TCED_CALIBRATION_TBD."""
    uncalibrated = DecisionEngine(calibrated=False)
    outcome, reason = uncalibrated.decide(
        {"status": "success", "prediction": "GENUINE"},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome == FinalOutcome.INCONCLUSIVE
    assert reason == ReasonCode.TCED_CALIBRATION_TBD.value


def test_missing_evidence_never_silently_genuine():
    """Partial evidence must NEVER silently classify as GENUINE."""
    # Acoustic GENUINE, RIAS missing
    outcome1, reason1 = decide(
        {"status": "success", "prediction": "GENUINE"},
        {"status": "error", "risk_level": None},
    )
    assert outcome1 == FinalOutcome.INCONCLUSIVE
    assert outcome1 != FinalOutcome.GENUINE
    assert reason1 == ReasonCode.RIAS_EVIDENCE_UNAVAILABLE.value

    # RIAS LOW, Acoustic missing
    outcome2, reason2 = decide(
        {"status": "error", "prediction": None},
        {"status": "success", "risk_level": "LOW"},
    )
    assert outcome2 == FinalOutcome.INCONCLUSIVE
    assert outcome2 != FinalOutcome.GENUINE
    assert reason2 == ReasonCode.ACOUSTIC_EVIDENCE_UNAVAILABLE.value
