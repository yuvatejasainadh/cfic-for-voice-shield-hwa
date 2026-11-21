"""Decision Engine for CFIC API Gateway.

Synthesizes acoustic spoof evidence with semantic conversational risk to
produce a deterministic final verdict: GENUINE, FAKE, SUSPICIOUS, or INCONCLUSIVE.

Locked Safety Rules:
1. Rule A & D: Acoustic FAKE always yields FAKE (Reason: ACOUSTIC_SPOOF_DETECTED).
   Acoustic spoof evidence must never be downgraded by RIAS reporting LOW or HIGH.
2. Rule H: If semantic risk analysis returns an invalid/unrecognized risk level, it must never
   be defaulted to LOW; it returns INCONCLUSIVE (Reason: INVALID_RIAS_RISK_LEVEL).
3. Rule C & E: If acoustic is GENUINE or UNAVAILABLE, and semantic risk returns HIGH,
   the outcome is SUSPICIOUS (Reason: HIGH_RIAS_OVERRIDE).
4. Rule B: If acoustic returns GENUINE and semantic risk returns LOW or MEDIUM,
   the outcome is GENUINE (Reason: ACOUSTIC_GENUINE_CONFIRMED).
5. Rule G: If both evidence sources are unavailable/failed, return INCONCLUSIVE
   (Reason: INSUFFICIENT_EVIDENCE).
6. Rule F: If acoustic detector is unavailable and semantic risk is LOW or MEDIUM,
   return INCONCLUSIVE (Reason: ACOUSTIC_EVIDENCE_UNAVAILABLE).
   LOW semantic risk is not proof of authentic speech.
"""

from typing import Any
from app.schemas.analysis import FinalOutcome, RiskLevel


class DecisionEngine:
    """Fuses multi-modal evidence streams into final decision verdicts."""

    def __init__(self, calibrated: bool = True):
        self.calibrated = calibrated

    def decide(
        self,
        spoof_res: dict[str, Any] | None,
        risk_res: dict[str, Any] | None,
    ) -> tuple[FinalOutcome, str]:
        """Synthesize acoustic detection and conversational risk.

        Args:
            spoof_res: Acoustic detector result dict (from PellaV2, VMC-South, or AASIST)
            risk_res: Semantic risk analysis result dict (from RIAS / Gemma)

        Returns:
            tuple[FinalOutcome, str]: (final_outcome, reason_code)
        """
        spoof = spoof_res or {}
        risk = risk_res or {}

        spoof_status = spoof.get("status")
        raw_pred = spoof.get("prediction")
        prediction = str(raw_pred).upper().strip() if raw_pred else None

        risk_status = risk.get("status")
        raw_risk = risk.get("risk_level")
        risk_level_val = (
            raw_risk.value
            if isinstance(raw_risk, RiskLevel)
            else (str(raw_risk).upper().strip() if raw_risk else None)
        )

        # RULE H: Validate semantic risk validity if reported success
        if risk_status == "success":
            valid_risk_levels = {r.value for r in RiskLevel}
            if risk_level_val not in valid_risk_levels:
                return FinalOutcome.INCONCLUSIVE, "INVALID_RIAS_RISK_LEVEL"

        # Check acoustic validity
        acoustic_ok = spoof_status == "success" and prediction in {"FAKE", "GENUINE"}

        # RULE A & D: Acoustic FAKE priority
        # When acoustic detector returns a valid FAKE classification,
        # spoof evidence must not be downgraded because RIAS reports LOW or HIGH.
        if acoustic_ok and prediction == "FAKE":
            return FinalOutcome.FAKE, "ACOUSTIC_SPOOF_DETECTED"

        # RULE C & E: HIGH RIAS Override
        # If RIAS is HIGH, and acoustic is either GENUINE or UNAVAILABLE, outcome is SUSPICIOUS.
        if risk_status == "success" and risk_level_val == RiskLevel.HIGH.value:
            return FinalOutcome.SUSPICIOUS, "HIGH_RIAS_OVERRIDE"

        # Optional uncalibrated legacy flag check
        if not self.calibrated and acoustic_ok and risk_status == "success":
            return FinalOutcome.INCONCLUSIVE, "TCED_CALIBRATION_TBD"

        # RULE B: Acoustic GENUINE + RIAS LOW/MEDIUM
        if (
            acoustic_ok
            and prediction == "GENUINE"
            and risk_status == "success"
            and risk_level_val in {RiskLevel.LOW.value, RiskLevel.MEDIUM.value}
        ):
            return FinalOutcome.GENUINE, "ACOUSTIC_GENUINE_CONFIRMED"

        # RULE G: Both evidence sources unavailable
        risk_ok = risk_status == "success" and risk_level_val in {
            RiskLevel.LOW.value,
            RiskLevel.MEDIUM.value,
        }
        if not acoustic_ok and not risk_ok:
            return FinalOutcome.INCONCLUSIVE, "INSUFFICIENT_EVIDENCE"

        # RULE F: Acoustic detector UNAVAILABLE + RIAS LOW (or partial evidence)
        # LOW semantic risk is not proof of authentic speech.
        return FinalOutcome.INCONCLUSIVE, "ACOUSTIC_EVIDENCE_UNAVAILABLE"


decision_engine = DecisionEngine(calibrated=True)
decide = decision_engine.decide  # export functional alias for backward compatibility
