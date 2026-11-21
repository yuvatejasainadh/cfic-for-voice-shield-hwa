"""Temporal and Contextual Evidence Decision (TCED) Policy Engine (Legacy Reference).

Combines PellaV2 audio deepfake detector evidence with RIAS (Risk Identification
and Assessment System) conversational risk analysis.

The TCED Complete Policy implements an authoritative 15-combination matrix across:
- Acoustic evidence: FAKE, GENUINE, UNAVAILABLE
- RIAS evidence: LOW, MEDIUM, HIGH, UNAVAILABLE, INVALID

Decision Priority Order:
1. Priority 1 — Acoustic FAKE:
   If acoustic evidence is FAKE, outcome is always FAKE (Reason: ACOUSTIC_SPOOF_DETECTED).
   RIAS must never override or downgrade acoustic spoof evidence.
2. Priority 2 — RIAS HIGH:
   If acoustic is not FAKE and RIAS is HIGH, outcome is SUSPICIOUS (Reason: HIGH_RIAS_OVERRIDE).
3. Priority 3 — Evidence Validity (RIAS INVALID):
   If acoustic is not FAKE and RIAS returns an invalid/unparseable risk level,
   outcome is INCONCLUSIVE (Reason: INVALID_RIAS_RISK_LEVEL). Invalid risk is never defaulted to LOW.
4. Priority 4 — Confirmed GENUINE:
   Only Acoustic GENUINE + RIAS LOW is allowed to produce GENUINE (Reason: ACOUSTIC_GENUINE_CONFIRMED).
5. Priority 5 — Remaining Evidence Combinations:
   - GENUINE + MEDIUM -> INCONCLUSIVE (Reason: EVIDENCE_CONFLICT)
   - GENUINE + UNAVAILABLE -> INCONCLUSIVE (Reason: RIAS_EVIDENCE_UNAVAILABLE)
   - UNAVAILABLE + LOW -> INCONCLUSIVE (Reason: ACOUSTIC_EVIDENCE_UNAVAILABLE)
   - UNAVAILABLE + MEDIUM -> INCONCLUSIVE (Reason: INSUFFICIENT_EVIDENCE)
   - UNAVAILABLE + UNAVAILABLE -> INCONCLUSIVE (Reason: INSUFFICIENT_EVIDENCE)
"""

from typing import Any
from app.schemas import FinalOutcome, ReasonCode, RiskLevel


def _classify_acoustic(spoof: dict[str, Any] | None) -> str:
    """Classify raw acoustic evidence into FAKE, GENUINE, or UNAVAILABLE."""
    if not spoof or not isinstance(spoof, dict):
        return "UNAVAILABLE"
    status = str(spoof.get("status", "")).lower().strip()
    if status != "success":
        return "UNAVAILABLE"
    raw_pred = spoof.get("prediction")
    if not raw_pred:
        return "UNAVAILABLE"
    pred = str(raw_pred).upper().strip()
    if pred == "FAKE":
        return "FAKE"
    if pred == "GENUINE":
        return "GENUINE"
    return "UNAVAILABLE"


def _classify_rias(risk: dict[str, Any] | None) -> str:
    """Classify raw RIAS risk evidence into LOW, MEDIUM, HIGH, UNAVAILABLE, or INVALID."""
    if not risk or not isinstance(risk, dict):
        return "UNAVAILABLE"
    status = str(risk.get("status", "")).lower().strip()
    if status != "success":
        return "UNAVAILABLE"
    raw_risk = risk.get("risk_level")
    if raw_risk is None:
        return "INVALID"
    if isinstance(raw_risk, RiskLevel):
        risk_str = raw_risk.value
    else:
        risk_str = str(raw_risk).upper().strip()
    if risk_str in {RiskLevel.LOW.value, RiskLevel.MEDIUM.value, RiskLevel.HIGH.value}:
        return risk_str
    return "INVALID"


def decide(pella: dict[str, Any] | None, rias: dict[str, Any] | None) -> tuple[FinalOutcome, str]:
    """Execute the TCED decision logic matching production DecisionEngine."""
    acoustic = _classify_acoustic(pella)
    rias_val = _classify_rias(rias)

    # PRIORITY 1: Acoustic FAKE
    if acoustic == "FAKE":
        return FinalOutcome.FAKE, ReasonCode.ACOUSTIC_SPOOF_DETECTED.value

    # PRIORITY 2: RIAS HIGH
    if rias_val == "HIGH":
        return FinalOutcome.SUSPICIOUS, ReasonCode.HIGH_RIAS_OVERRIDE.value

    # PRIORITY 3: RIAS INVALID
    if rias_val == "INVALID":
        return FinalOutcome.INCONCLUSIVE, ReasonCode.INVALID_RIAS_RISK_LEVEL.value

    # PRIORITY 4: Confirmed GENUINE
    if acoustic == "GENUINE" and rias_val == "LOW":
        return FinalOutcome.GENUINE, ReasonCode.ACOUSTIC_GENUINE_CONFIRMED.value

    # PRIORITY 5: Remaining combinations (All yield INCONCLUSIVE)
    if acoustic == "GENUINE":
        if rias_val == "MEDIUM":
            return FinalOutcome.INCONCLUSIVE, ReasonCode.EVIDENCE_CONFLICT.value
        if rias_val == "UNAVAILABLE":
            return FinalOutcome.INCONCLUSIVE, ReasonCode.RIAS_EVIDENCE_UNAVAILABLE.value

    if acoustic == "UNAVAILABLE":
        if rias_val == "LOW":
            return FinalOutcome.INCONCLUSIVE, ReasonCode.ACOUSTIC_EVIDENCE_UNAVAILABLE.value
        if rias_val in {"MEDIUM", "UNAVAILABLE"}:
            return FinalOutcome.INCONCLUSIVE, ReasonCode.INSUFFICIENT_EVIDENCE.value

    return FinalOutcome.INCONCLUSIVE, ReasonCode.INSUFFICIENT_EVIDENCE.value
