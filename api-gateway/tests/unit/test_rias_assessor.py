"""Unit tests for RIAS Stage 2 AskDrip LLM Assessor."""

import pytest
import httpx
from services.rias.assessor import assess_transcript
from services.rias.prompt import build_risk_assessment_prompt


class TestRiasAssessor:
    def test_build_prompt_includes_clean_transcript(self):
        prompt = build_risk_assessment_prompt("  Hello bank caller asking for OTP  ")
        assert "Hello bank caller asking for OTP" in prompt
        assert "<transcript>" in prompt

    @pytest.mark.asyncio
    async def test_empty_transcript_skipped(self):
        result = await assess_transcript("")
        assert result["status"] == "skipped"
        assert result["risk_level"] is None

    @pytest.mark.asyncio
    async def test_unconfigured_credentials_returns_not_configured(self):
        result = await assess_transcript("Sample text", base_url="", api_key="")
        assert result["status"] == "not_configured"
        assert result["risk_level"] is None

    @pytest.mark.asyncio
    async def test_successful_llm_response_parsing(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            return httpx.Response(
                status_code=200,
                json={
                    "risk_level": "HIGH",
                    "reason": "Caller aggressively demanding bank OTP credentials",
                },
                request=httpx.Request("POST", "http://test-llm"),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        result = await assess_transcript(
            "Give me your bank password immediately",
            base_url="http://test-llm",
            api_key="valid-test-key",
        )

        assert result["status"] == "success"
        assert result["risk_level"] == "HIGH"
        assert "OTP credentials" in result["reason"]

    @pytest.mark.asyncio
    async def test_malformed_llm_json_fails_safely_never_low(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            return httpx.Response(
                status_code=200,
                text="Not valid JSON at all",
                request=httpx.Request("POST", "http://test-llm"),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        result = await assess_transcript(
            "Some text",
            base_url="http://test-llm",
            api_key="valid-test-key",
        )

        # Safety requirement: Failure MUST NOT silently become LOW
        assert result["status"] == "error"
        assert result["risk_level"] is None
        assert "parse structured JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_risk_level_from_llm_fails_safely(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            return httpx.Response(
                status_code=200,
                json={"risk_level": "EXTREME_DANGER", "reason": "unrecognized level"},
                request=httpx.Request("POST", "http://test-llm"),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        result = await assess_transcript(
            "Some text",
            base_url="http://test-llm",
            api_key="valid-test-key",
        )

        # Safety requirement: Unrecognized risk level MUST NOT silently become LOW
        assert result["status"] == "error"
        assert result["risk_level"] is None
        assert "Invalid risk_level" in result["error"]
