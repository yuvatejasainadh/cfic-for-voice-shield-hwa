"""Temporal and Contextual Evidence Decision (TCED) Policy Engine.

Combines PellaV2 audio deepfake detector evidence with RIAS (Risk Identification
and Assessment System) conversational risk analysis.

Architecture Rules:
1. Locked Rule: If RIAS returns HIGH, the final outcome must be SUSPICIOUS,
   regardless of the PellaV2 result (Reason: HIGH_RIAS_OVERRIDE).
2. If both analysis paths fail or provide insufficient evidence, return INCONCLUSIVE
   (Reason: INSUFFICIENT_EVIDENCE).
3. If partial evidence is present (e.g. only one path succeeds), the system must NOT
   silently classify as GENUINE; it returns INCONCLUSIVE (Reason: PARTIAL_EVIDENCE_POLICY_TBD).
4. If RIAS returns an invalid or unrecognized risk level, it must never be defaulted
   to LOW; it returns INCONCLUSIVE (Reason: INVALID_RIAS_RISK_LEVEL).
5. Uncalibrated weights and thresholds: Until Varun verifies the exact PellaV2 score
   polarity and Sainadh validates the hybrid calibration weights against empirical data,
   full evidence returns INCONCLUSIVE (Reason: TCED_CALIBRATION_TBD).
"""

from typing import Any
from app.schemas import FinalOutcome, RiskLevel


# Placeholders for future empirical calibration (documented in docs/DECISIONS_AND_TBDS.md)
# These MUST remain explicitly marked as TBD until team alignment and validation.
TCED_CALIBRATED: bool = False
TCED_PELLA_WEIGHT: float = 0.60  # Subject to empirical tuning
TCED_RIAS_WEIGHT: float = 0.40   # Subject to empirical tuning
TCED_FAKE_THRESHOLD: float = 0.75
TCED_GENUINE_THRESHOLD: float = 0.25


def decide(pella: dict[str, Any] | None, rias: dict[str, Any] | None) -> tuple[FinalOutcome, str]:
    """Execute the TCED decision logic.

    Args:
        pella: Response dictionary from PellaV2 detector client.
               Expected keys: "status", "prediction", "score", "model_version"
        rias: Response dictionary from RIAS assessor client.
              Expected keys: "status", "risk_level"

    Returns:
        tuple[FinalOutcome, str]: (Final outcome enum, Machine-readable reason code)
    """
    pella = pella or {}
    rias = rias or {}

    pella_status = pella.get("status")
    rias_status = rias.get("status")
    rias_risk = rias.get("risk_level")

    # RULE 1 (MANDATORY): RIAS HIGH risk ALWAYS overrides to SUSPICIOUS
    if rias_status == "success" and rias_risk == RiskLevel.HIGH.value:
        return FinalOutcome.SUSPICIOUS, "HIGH_RIAS_OVERRIDE"

    # RULE 2: Validate RIAS risk level validity if RIAS reported success
    if rias_status == "success":
        valid_risk_levels = {r.value for r in RiskLevel}
        if rias_risk not in valid_risk_levels:
            # Unknown risk level must NEVER default to LOW
            return FinalOutcome.INCONCLUSIVE, "INVALID_RIAS_RISK_LEVEL"

    # RULE 3: Check component success
    pella_ok = pella_status == "success" and pella.get("prediction") is not None
    rias_ok = rias_status == "success" and rias_risk in {
        RiskLevel.LOW.value,
        RiskLevel.MEDIUM.value,
    }

    # If both components failed to provide valid output
    if not pella_ok and not rias_ok:
        return FinalOutcome.INCONCLUSIVE, "INSUFFICIENT_EVIDENCE"

    # If only one component succeeded (partial evidence)
    if not pella_ok or not rias_ok:
        return FinalOutcome.INCONCLUSIVE, "PARTIAL_EVIDENCE_POLICY_TBD"

    # RULE 4: Both components succeeded with LOW or MEDIUM risk
    if not TCED_CALIBRATED:
        # Score calibration and threshold weighting are pending empirical validation
        return FinalOutcome.INCONCLUSIVE, "TCED_CALIBRATION_TBD"

    # Calibration logic (active only once TCED_CALIBRATED is enabled with verified parameters)
    # Kept as clean reference implementation for the team:
    return _evaluate_calibrated_decision(pella, rias)


def _evaluate_calibrated_decision(pella: dict[str, Any], rias: dict[str, Any]) -> tuple[FinalOutcome, str]:
    """Internal helper for calibrated scoring once validated."""
    # Note: Varun must confirm whether higher score means fake or genuine in PellaV2 artifact
    prediction = str(pella.get("prediction", "")).upper()
    rias_risk = rias.get("risk_level")

    if prediction == "FAKE":
        return FinalOutcome.FAKE, "PELLA_DETECTED_SYNTHETIC"
    if prediction == "GENUINE" and rias_risk == RiskLevel.LOW.value:
        return FinalOutcome.GENUINE, "VERIFIED_GENUINE_LOW_RISK"
    if rias_risk == RiskLevel.MEDIUM.value:
        return FinalOutcome.SUSPICIOUS, "MEDIUM_RISK_CAUTION"

    return FinalOutcome.INCONCLUSIVE, "UNCERTAIN_EVIDENCE"
