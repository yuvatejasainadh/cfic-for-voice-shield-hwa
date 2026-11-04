"""RIAS service compatibility package."""

from services.rias.assessor import assess_transcript
from services.rias.prompt import RIAS_SYSTEM_PROMPT, build_risk_assessment_prompt

__all__ = ["RIAS_SYSTEM_PROMPT", "assess_transcript", "build_risk_assessment_prompt"]
