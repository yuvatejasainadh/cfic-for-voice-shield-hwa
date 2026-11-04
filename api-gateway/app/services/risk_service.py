"""Application-level conversational risk assessment service (RIAS integration).

Evaluates conversational transcripts to produce a strict categorical risk:
LOW, MEDIUM, or HIGH.
"""

import json
import time
from typing import Any
import httpx
from app.core.config import settings
from app.schemas.analysis import RiskLevel

RIAS_SYSTEM_PROMPT = """You are RIAS (Risk Identification and Assessment System), an advanced conversational security analyst integrated into CloudForge Intelligence Core (CFIC).

Your task is to analyze an audio transcript from an incoming telephone call to detect social engineering, urgency manipulation, extortion, financial coercion, impersonation (e.g. bank official, law enforcement, utility provider, family member in distress), and credential theft.

Evaluate the conversational risk into exactly one of three categories:
- LOW: Normal personal, commercial, or informational conversation with no indicators of fraud, coercion, urgent financial demands, or credential phishing.
- MEDIUM: Ambiguous, unsolicited, mildly pushy, or unverified claims requesting callbacks, personal details, or unusual actions, but without extreme coercion or immediate irreversible fund transfers.
- HIGH: Direct social engineering, urgent requests for OTP/passwords/wire transfers/gift cards, arrest threats, legal intimidation, emergency impersonation, or high-risk fraudulent instructions.

You must respond ONLY with a strict JSON object with no additional text or Markdown formatting:
{
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "reason": "Concise summary of rationale (under 25 words)"
}
"""


def build_risk_assessment_prompt(transcript: str) -> str:
    """Format user prompt for conversational risk analysis."""
    clean_text = transcript.strip()
    return f"Analyze the following call transcript for social engineering and security risk:\n\n<transcript>\n{clean_text}\n</transcript>"


class RiskService:
    """Service governing semantic fraud risk assessment."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (base_url if base_url is not None else settings.askdrip_llm_base_url or "").rstrip("/")
        self.api_key = api_key if api_key is not None else settings.askdrip_llm_api_key
        self.timeout = timeout if timeout is not None else settings.request_timeout_seconds

    async def assess_transcript(
        self,
        transcript: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Assess transcript risk via the hosted LLM API.

        Safety Guarantee:
        If the LLM call fails, times out, or produces unparseable/unknown risk levels,
        this function returns status='error' and risk_level=None. It NEVER assumes LOW.
        """
        start = time.perf_counter()

        clean_text = (transcript or "").strip()
        if not clean_text:
            return {
                "status": "skipped",
                "risk_level": None,
                "reason": "Empty transcript",
                "error": None,
                "latency_ms": 0.0,
            }

        url = (base_url if base_url is not None else self.base_url).rstrip("/")
        key = api_key if api_key is not None else self.api_key
        to = timeout if timeout is not None else self.timeout

        if not url or not key:
            return {
                "status": "not_configured",
                "risk_level": None,
                "reason": None,
                "error": "AskDrip LLM service URL or API key is not configured",
                "latency_ms": 0.0,
            }

        payload = {
            "model": settings.askdrip_llm_model,
            "messages": [
                {"role": "system", "content": RIAS_SYSTEM_PROMPT},
                {"role": "user", "content": build_risk_assessment_prompt(clean_text)},
            ],
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "CFIC-VoiceShield-RIAS/0.1.0",
        }

        try:
            async with httpx.AsyncClient(timeout=to) as client:
                response = await client.post(url, json=payload, headers=headers)
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

                if response.status_code != 200:
                    return {
                        "status": "error",
                        "risk_level": None,
                        "reason": None,
                        "error": f"AskDrip LLM returned HTTP {response.status_code}",
                        "latency_ms": elapsed_ms,
                    }

                raw_body = response.text.strip()
                try:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"]["content"]
                        parsed = json.loads(content)
                    elif "risk_level" in data:
                        parsed = data
                    else:
                        parsed = json.loads(raw_body)
                except Exception:
                    return {
                        "status": "error",
                        "risk_level": None,
                        "reason": None,
                        "error": "Failed to parse structured JSON from LLM response",
                        "latency_ms": elapsed_ms,
                    }

                risk_level_str = str(parsed.get("risk_level", "")).upper()
                valid_risks = {r.value for r in RiskLevel}

                if risk_level_str not in valid_risks:
                    return {
                        "status": "error",
                        "risk_level": None,
                        "reason": None,
                        "error": f"Invalid risk_level '{risk_level_str}' returned by LLM",
                        "latency_ms": elapsed_ms,
                    }

                return {
                    "status": "success",
                    "risk_level": risk_level_str,
                    "reason": parsed.get("reason", "Assessment completed"),
                    "error": None,
                    "latency_ms": elapsed_ms,
                }
        except httpx.TimeoutException:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "status": "timeout",
                "risk_level": None,
                "reason": None,
                "error": "AskDrip LLM request timed out",
                "latency_ms": elapsed_ms,
            }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "status": "error",
                "risk_level": None,
                "reason": None,
                "error": f"AskDrip LLM request failed: {type(exc).__name__}",
                "latency_ms": elapsed_ms,
            }


risk_service = RiskService()
assess_transcript = risk_service.assess_transcript
