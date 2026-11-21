"""Temporal and Contextual Evidence Decision (TCED) Policy Engine (Legacy Reference).

Combines PellaV2 audio deepfake detector evidence with RIAS (Risk Identification
and Assessment System) conversational risk analysis.

Architecture Rules:
1. Rule A & D: Acoustic FAKE always yields FAKE (Reason: ACOUSTIC_SPOOF_DETECTED).
   Acoustic spoof evidence must never be downgraded by RIAS reporting LOW or HIGH.
2. Rule H: If RIAS returns an invalid or unrecognized risk level, it must never be defaulted
   to LOW; it returns INCONCLUSIVE (Reason: INVALID_RIAS_RISK_LEVEL).
3. Rule C & E: If acoustic is GENUINE or UNAVAILABLE, and RIAS returns HIGH,
   the outcome is SUSPICIOUS (Reason: HIGH_RIAS_OVERRIDE).
4. Rule B: If acoustic returns GENUINE and RIAS returns LOW or MEDIUM,
   the outcome is GENUINE (Reason: ACOUSTIC_GENUINE_CONFIRMED).
5. Rule G: If both analysis paths fail or provide insufficient evidence, return INCONCLUSIVE
   (Reason: INSUFFICIENT_EVIDENCE).
6. Rule F: If acoustic detector is unavailable and RIAS returns LOW, return INCONCLUSIVE
   (Reason: ACOUSTIC_EVIDENCE_UNAVAILABLE).
"""

from typing import Any
from app.schemas import FinalOutcome, RiskLevel


def decide(pella: dict[str, Any] | None, rias: dict[str, Any] | None) -> tuple[FinalOutcome, str]:
    """Execute the TCED decision logic."""
    pella = pella or {}
    rias = rias or {}

    pella_status = pella.get("status")
    raw_pred = pella.get("prediction")
    prediction = str(raw_pred).upper().strip() if raw_pred else None

    rias_status = rias.get("status")
    raw_risk = rias.get("risk_level")
    risk_level_val = (
        raw_risk.value
        if isinstance(raw_risk, RiskLevel)
        else (str(raw_risk).upper().strip() if raw_risk else None)
    )

    # RULE H: Validate RIAS risk level validity if RIAS reported success
    if rias_status == "success":
        valid_risk_levels = {r.value for r in RiskLevel}
        if risk_level_val not in valid_risk_levels:
            return FinalOutcome.INCONCLUSIVE, "INVALID_RIAS_RISK_LEVEL"

    acoustic_ok = pella_status == "success" and prediction in {"FAKE", "GENUINE"}

    # RULE A & D: Acoustic FAKE priority
    if acoustic_ok and prediction == "FAKE":
        return FinalOutcome.FAKE, "ACOUSTIC_SPOOF_DETECTED"

    # RULE C & E: RIAS HIGH override
    if rias_status == "success" and risk_level_val == RiskLevel.HIGH.value:
        return FinalOutcome.SUSPICIOUS, "HIGH_RIAS_OVERRIDE"

    # RULE B: Acoustic GENUINE + RIAS LOW/MEDIUM
    if (
        acoustic_ok
        and prediction == "GENUINE"
        and rias_status == "success"
        and risk_level_val in {RiskLevel.LOW.value, RiskLevel.MEDIUM.value}
    ):
        return FinalOutcome.GENUINE, "ACOUSTIC_GENUINE_CONFIRMED"

    # RULE G: Both components failed
    risk_ok = rias_status == "success" and risk_level_val in {
        RiskLevel.LOW.value,
        RiskLevel.MEDIUM.value,
    }
    if not acoustic_ok and not risk_ok:
        return FinalOutcome.INCONCLUSIVE, "INSUFFICIENT_EVIDENCE"

    # RULE F: Acoustic detector UNAVAILABLE + RIAS LOW
    return FinalOutcome.INCONCLUSIVE, "ACOUSTIC_EVIDENCE_UNAVAILABLE"
