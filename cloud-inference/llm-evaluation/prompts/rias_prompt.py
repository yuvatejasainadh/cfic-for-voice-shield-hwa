"""Prompt definition and parsing utilities for RIAS semantic-risk evaluation."""

from pathlib import Path
import json
import re

PROMPTS_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "rias_system_prompt.txt"

with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    RIAS_SYSTEM_PROMPT = f.read().strip()


def build_user_prompt(transcript: str) -> str:
    """Format user prompt for conversational risk analysis."""
    clean_text = (transcript or "").strip()
    return f"Analyze the following call transcript for social engineering and security risk:\n\n<transcript>\n{clean_text}\n</transcript>"


def parse_rias_response(raw_response: str) -> tuple[str, str, bool]:
    """Parse raw LLM completion into (risk_level, reason, is_valid).
    
    Returns:
        tuple[str, str, bool]: (risk_level, reason, is_valid)
        If invalid, risk_level is 'INCONCLUSIVE' and is_valid is False.
    """
    text = (raw_response or "").strip()
    if not text:
        return "INCONCLUSIVE", "Empty model response", False

    # Strip markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
        risk = str(data.get("risk_level", "")).strip().upper()
        reason = str(data.get("reason", "Assessment completed")).strip()
        if risk in {"LOW", "MEDIUM", "HIGH"}:
            return risk, reason, True
        return "INCONCLUSIVE", f"Invalid risk level value: {risk}", False
    except Exception:
        # Fallback regex search for strict JSON object or risk_level pattern
        match = re.search(r'["\']risk_level["\']\s*:\s*["\'](LOW|MEDIUM|HIGH)["\']', text, re.IGNORECASE)
        if match:
            risk = match.group(1).upper()
            return risk, "Parsed via regex fallback", True
        return "INCONCLUSIVE", f"Malformed JSON response: {text[:80]}", False
