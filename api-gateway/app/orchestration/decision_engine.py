"""Decision Engine for CFIC API Gateway.

Synthesizes acoustic spoof evidence with semantic conversational risk to
produce a deterministic final verdict: GENUINE, FAKE, SUSPICIOUS, or INCONCLUSIVE.

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
from app.schemas.analysis import FinalOutcome, ReasonCode, RiskLevel


class DecisionEngine:
    """Fuses multi-modal evidence streams into final decision verdicts."""

    def __init__(self, calibrated: bool = True):
        self.calibrated = calibrated

    @staticmethod
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

    @staticmethod
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
        acoustic = self._classify_acoustic(spoof_res)
        rias = self._classify_rias(risk_res)

        # PRIORITY 1: Acoustic FAKE
        # Absolute precedence: spoof evidence must never be downgraded by RIAS
        if acoustic == "FAKE":
            return FinalOutcome.FAKE, ReasonCode.ACOUSTIC_SPOOF_DETECTED.value

        # PRIORITY 2: RIAS HIGH
        # High conversational/semantic risk overrides GENUINE or UNAVAILABLE acoustic evidence
        if rias == "HIGH":
            return FinalOutcome.SUSPICIOUS, ReasonCode.HIGH_RIAS_OVERRIDE.value

        # PRIORITY 3: RIAS INVALID
        # Invalid risk data must never be defaulted to LOW or treated as authentic
        if rias == "INVALID":
            return FinalOutcome.INCONCLUSIVE, ReasonCode.INVALID_RIAS_RISK_LEVEL.value

        # Uncalibrated engine check (for pre-calibration staging environments)
        if (
            not self.calibrated
            and acoustic == "GENUINE"
            and rias in {"LOW", "MEDIUM"}
        ):
            return FinalOutcome.INCONCLUSIVE, ReasonCode.TCED_CALIBRATION_TBD.value

        # PRIORITY 4: Confirmed GENUINE
        # Only Acoustic GENUINE + RIAS LOW is allowed to produce GENUINE
        if acoustic == "GENUINE" and rias == "LOW":
            return FinalOutcome.GENUINE, ReasonCode.ACOUSTIC_GENUINE_CONFIRMED.value

        # PRIORITY 5: Remaining combinations (All yield INCONCLUSIVE)
        if acoustic == "GENUINE":
            if rias == "MEDIUM":
                return FinalOutcome.INCONCLUSIVE, ReasonCode.EVIDENCE_CONFLICT.value
            if rias == "UNAVAILABLE":
                return FinalOutcome.INCONCLUSIVE, ReasonCode.RIAS_EVIDENCE_UNAVAILABLE.value

        if acoustic == "UNAVAILABLE":
            if rias == "LOW":
                return FinalOutcome.INCONCLUSIVE, ReasonCode.ACOUSTIC_EVIDENCE_UNAVAILABLE.value
            if rias in {"MEDIUM", "UNAVAILABLE"}:
                return FinalOutcome.INCONCLUSIVE, ReasonCode.INSUFFICIENT_EVIDENCE.value

        # Unreachable fail-safe
        return FinalOutcome.INCONCLUSIVE, ReasonCode.INSUFFICIENT_EVIDENCE.value


decision_engine = DecisionEngine(calibrated=True)
decide = decision_engine.decide  # export functional alias for backward compatibility
