"""Robustness and edge-case evaluation suite for RIAS LLM candidates."""

import os
import sys
import json
import time
from pathlib import Path
from llama_cpp import Llama

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EVAL_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from prompts.rias_prompt import RIAS_SYSTEM_PROMPT, build_user_prompt, parse_rias_response

ROBUSTNESS_CASES = [
    {
        "name": "empty_transcript",
        "transcript": "",
        "expected_behavior": "Must not fail or assume LOW; caller handles empty or LLM produces valid/inconclusive"
    },
    {
        "name": "whitespace_only",
        "transcript": "     \n\t   ",
        "expected_behavior": "Must not crash or hallucinate false risk"
    },
    {
        "name": "very_short_benign",
        "transcript": "Hello, thank you.",
        "expected_risk": "LOW"
    },
    {
        "name": "very_short_urgent_scam",
        "transcript": "Share OTP immediately or police will arrest you.",
        "expected_risk": "HIGH"
    },
    {
        "name": "repeated_words_probe",
        "transcript": "bank bank bank bank account urgent block verify otp otp otp otp now now now",
        "expected_risk": "HIGH"
    },
    {
        "name": "garbled_acoustic_noise",
        "transcript": "kzzzt ... static ... [unintelligible] 8943729 #@$#% beep buzz ...",
        "expected_behavior": "Should be classified as LOW or MEDIUM, or fail safely to INCONCLUSIVE. Never false HIGH."
    },
    {
        "name": "code_switched_hinglish_high",
        "transcript": "Sir aapka SBI account deactivate hone wala hai, turant phone pe aaya hua six-digit OTP number bataiye urgently.",
        "expected_risk": "HIGH"
    },
    {
        "name": "code_switched_tenglish_high",
        "transcript": "Mee bank account block avtundi, ventane mee debit card details and OTP share cheyandi urgently.",
        "expected_risk": "HIGH"
    },
    {
        "name": "long_benign_transcript",
        "transcript": (
            "Good morning Dr. Sharma, I am calling to follow up on the blood test results from last Wednesday. "
            "The technician mentioned that the hemoglobin count was slightly lower than normal, but the cholesterol "
            "and liver function markers were completely within range. I also wanted to ask about the dietary recommendations "
            "you gave regarding iron-rich vegetables like spinach and lentils. Should I continue with the multivitamin supplement "
            "for another four weeks, or do you recommend scheduling an in-person consultation next Monday afternoon? "
            "I will be traveling to Bangalore later this month for a family wedding, so having the prescription updated "
            "before my departure would be very helpful. Please let me know what time works best for your clinic staff. Thank you!"
        ),
        "expected_risk": "LOW"
    },
    {
        "name": "benign_containing_sensitive_keywords",
        "transcript": "I had to report my stolen wallet to the police station yesterday, and then I called the bank customer care to block my lost cards and request a new password.",
        "expected_risk": "LOW"
    }
]


def run_robustness_tests(model_name: str, model_path: Path) -> dict:
    print(f"\n=======================================================")
    print(f"RUNNING ROBUSTNESS TESTS: {model_name}")
    print(f"=======================================================")

    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )

    results = []
    passed_count = 0

    for test in ROBUSTNESS_CASES:
        t_name = test["name"]
        transcript = test["transcript"]
        expected_risk = test.get("expected_risk")

        user_prompt = build_user_prompt(transcript)
        messages = [
            {"role": "system", "content": RIAS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        t0 = time.perf_counter()
        completion = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=64,
            stop=["\n\n", "}"]
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        raw_text = completion["choices"][0]["message"]["content"] or ""
        if "{" in raw_text and not raw_text.strip().endswith("}"):
            raw_text = raw_text.strip() + "}"

        pred_risk, reason, is_valid = parse_rias_response(raw_text)

        passed = False
        if expected_risk:
            passed = (pred_risk == expected_risk)
        else:
            # General stability test
            passed = is_valid or pred_risk in {"LOW", "MEDIUM", "INCONCLUSIVE"}

        if passed:
            passed_count += 1

        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] {t_name:35s} | Pred: {pred_risk:12s} | Exp: {str(expected_risk):8s} | Latency: {latency_ms:.1f}ms")

        results.append({
            "test_name": t_name,
            "transcript_preview": transcript[:60],
            "expected_risk": expected_risk,
            "predicted_risk": pred_risk,
            "is_valid": is_valid,
            "reason": reason,
            "latency_ms": latency_ms,
            "passed": passed
        })

    summary = {
        "model_name": model_name,
        "total_tests": len(ROBUSTNESS_CASES),
        "passed_tests": passed_count,
        "pass_rate": round(passed_count / len(ROBUSTNESS_CASES), 4),
        "details": results
    }

    del llm
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_robustness.py <model_name> <model_path>")
        sys.exit(1)

    m_name = sys.argv[1]
    m_path = Path(sys.argv[2])
    out = run_robustness_tests(m_name, m_path)
    print(f"\nRobustness score: {out['passed_tests']}/{out['total_tests']} ({out['pass_rate']*100:.1f}%)")
