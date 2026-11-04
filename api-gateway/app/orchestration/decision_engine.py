"""Decision Engine for CFIC API Gateway.

Synthesizes acoustic spoof evidence with semantic conversational risk to
produce a deterministic final verdict: GENUINE, FAKE, SUSPICIOUS, or INCONCLUSIVE.

Locked Safety Rules:
1. If semantic risk analysis returns HIGH, the final outcome MUST be SUSPICIOUS
   (Reason: HIGH_RIAS_OVERRIDE), regardless of acoustic detector verdict.
2. If semantic risk analysis returns an invalid/unrecognized risk level, it must never
   be defaulted to LOW; it returns INCONCLUSIVE (Reason: INVALID_RIAS_RISK_LEVEL).
3. If components fail, time out, or provide missing/partial evidence, the system
   returns INCONCLUSIVE. It never silently classifies an error as GENUINE.
"""

from typing import Any
from app.schemas.analysis import FinalOutcome, RiskLevel


class DecisionEngine:
    """Fuses multi-modal evidence streams into final decision verdicts."""

    def __init__(self, calibrated: bool = False):
        self.calibrated = calibrated

    def decide(
        self,
        spoof_res: dict[str, Any] | None,
        risk_res: dict[str, Any] | None,
    ) -> tuple[FinalOutcome, str]:
        """Synthesize acoustic detection and conversational risk.

        Args:
            spoof_res: Acoustic detector result dict (from VMC-South, AASIST, or legacy Pella)
            risk_res: Semantic risk analysis result dict (from RIAS / RiskService)

        Returns:
            tuple[FinalOutcome, str]: (final_outcome, reason_code)
        """
        spoof = spoof_res or {}
        risk = risk_res or {}

        spoof_status = spoof.get("status")
        risk_status = risk.get("status")
        raw_risk = risk.get("risk_level")
        risk_level_val = raw_risk.value if isinstance(raw_risk, RiskLevel) else (str(raw_risk) if raw_risk else None)

        # RULE 1: Locked Safety Override - HIGH risk forces SUSPICIOUS
        if risk_status == "success" and risk_level_val == RiskLevel.HIGH.value:
            return FinalOutcome.SUSPICIOUS, "HIGH_RIAS_OVERRIDE"

        # RULE 2: Validate semantic risk validity if reported success
        if risk_status == "success":
            valid_risk_levels = {r.value for r in RiskLevel}
            if risk_level_val not in valid_risk_levels:
                return FinalOutcome.INCONCLUSIVE, "INVALID_RIAS_RISK_LEVEL"

        # RULE 3: Check component success
        spoof_ok = spoof_status == "success" and spoof.get("prediction") is not None
        risk_ok = risk_status == "success" and risk_level_val in {
            RiskLevel.LOW.value,
            RiskLevel.MEDIUM.value,
        }

        # If both components failed to provide valid output
        if not spoof_ok and not risk_ok:
            return FinalOutcome.INCONCLUSIVE, "INSUFFICIENT_EVIDENCE"

        # If only one component succeeded (partial evidence)
        if not spoof_ok or not risk_ok:
            return FinalOutcome.INCONCLUSIVE, "PARTIAL_EVIDENCE_POLICY_TBD"

        # RULE 5: Calibrated score fusion or TBD placeholder
        if not self.calibrated:
            return FinalOutcome.INCONCLUSIVE, "TCED_CALIBRATION_TBD"

        return self._evaluate_calibrated_fusion(spoof, risk)

    def _evaluate_calibrated_fusion(
        self, spoof: dict[str, Any], risk: dict[str, Any]
    ) -> tuple[FinalOutcome, str]:
        """Calibrated mathematical fusion once empirical weights are verified."""
        prediction = (spoof.get("prediction") or "").upper()
        if prediction == "FAKE":
            return FinalOutcome.FAKE, "ACOUSTIC_SPOOF_DETECTED"
        elif prediction == "GENUINE":
            return FinalOutcome.GENUINE, "ACOUSTIC_GENUINE_CONFIRMED"
        return FinalOutcome.INCONCLUSIVE, "UNRESOLVED_SCORE"


decision_engine = DecisionEngine()
decide = decision_engine.decide  # export functional alias for backward compatibility
