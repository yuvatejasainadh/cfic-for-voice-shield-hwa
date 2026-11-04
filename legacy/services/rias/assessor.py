"""RIAS Stage 2: Hosted AskDrip LLM Risk Assessor client.

Evaluates conversational transcripts to produce a strict categorical risk:
LOW, MEDIUM, or HIGH.
"""

import json
import time
from typing import Any
import httpx
from app.core.config import settings
from app.schemas import RiskLevel
from services.rias.prompt import RIAS_SYSTEM_PROMPT, build_risk_assessment_prompt


async def assess_transcript(
    transcript: str,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Assess transcript risk via the hosted AskDrip LLM API.

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

    raw_url = base_url if base_url is not None else (settings.askdrip_llm_base_url or "")
    url = raw_url.rstrip("/") if raw_url else ""
    key = api_key if api_key is not None else settings.askdrip_llm_api_key
    to = timeout if timeout is not None else settings.request_timeout_seconds

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
