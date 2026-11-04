"""Unit tests for the RIAS Adapter prototype and prompt parsing logic."""

import sys
from pathlib import Path
EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))

import pytest
from prompts.rias_prompt import parse_rias_response, build_user_prompt, RIAS_SYSTEM_PROMPT
from rias_adapter import RIASRequest, RIASResponse, RIASAdapter


def test_system_prompt_present_and_well_formed():
    assert len(RIAS_SYSTEM_PROMPT) > 200
    assert "LOW" in RIAS_SYSTEM_PROMPT
    assert "MEDIUM" in RIAS_SYSTEM_PROMPT
    assert "HIGH" in RIAS_SYSTEM_PROMPT
    assert "risk_level" in RIAS_SYSTEM_PROMPT


def test_build_user_prompt():
    prompt = build_user_prompt("  Hello, this is a test.  ")
    assert "<transcript>\nHello, this is a test.\n</transcript>" in prompt


def test_parse_valid_json_response():
    raw = '{"risk_level": "HIGH", "reason": "Immediate demand for OTP"}'
    risk, reason, valid = parse_rias_response(raw)
    assert valid is True
    assert risk == "HIGH"
    assert "demand for OTP" in reason


def test_parse_markdown_fenced_json():
    raw = '```json\n{\n  "risk_level": "LOW",\n  "reason": "Normal casual chat"\n}\n```'
    risk, reason, valid = parse_rias_response(raw)
    assert valid is True
    assert risk == "LOW"
    assert "casual chat" in reason


def test_parse_regex_fallback_on_unclosed_json():
    raw = 'Based on analysis: {"risk_level": "MEDIUM", "reason": "Ambiguous unverified call'
    risk, reason, valid = parse_rias_response(raw)
    assert valid is True
    assert risk == "MEDIUM"


def test_parse_invalid_risk_level_produces_inconclusive():
    raw = '{"risk_level": "CRITICAL_FRAUD", "reason": "Severe attack"}'
    risk, reason, valid = parse_rias_response(raw)
    assert valid is False
    assert risk == "INCONCLUSIVE"


def test_parse_empty_string_produces_inconclusive():
    risk, reason, valid = parse_rias_response("")
    assert valid is False
    assert risk == "INCONCLUSIVE"


def test_empty_transcript_produces_skipped_and_inconclusive():
    # Adapter does not need to load actual LLM weights for empty transcript test
    adapter = RIASAdapter(
        model_path="dummy_path.gguf",
        model_id="test/model"
    )
    req = RIASRequest(
        session_id="sess_12345",
        chunk_index=2,
        timestamp_start_s=20.0,
        timestamp_end_s=30.0,
        transcript="   \n\t   "
    )
    res = adapter.assess(req)
    assert res.status == "skipped"
    assert res.risk_level == "INCONCLUSIVE"
    assert res.session_id == "sess_12345"
    assert res.chunk_index == 2
    assert res.timestamp_start_s == 20.0
    assert res.timestamp_end_s == 30.0
