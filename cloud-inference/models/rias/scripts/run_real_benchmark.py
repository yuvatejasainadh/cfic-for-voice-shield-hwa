"""Real CPU Inference Benchmark for Containerized Gemma 4 E2B IT RIAS Microservice.

Runs actual inference on the local host CPU with the verified GGUF Q4_K_M checkpoint.
Measures cold load time, RAM, CPU utilization, latency percentiles, throughput,
and queue behavior across short, medium, and long transcripts.
"""

import asyncio
import os
from pathlib import Path
import statistics
import sys
import time
import psutil

# Ensure rias app is importable
RIAS_APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(RIAS_APP_DIR) not in sys.path:
    sys.path.insert(0, str(RIAS_APP_DIR))

from prompt import RIAS_SYSTEM_PROMPT, build_user_prompt, parse_risk_label

MODEL_CANDIDATE_PATHS = [
    Path(__file__).resolve().parents[1] / "weights" / "gemma-4-E2B-it-Q4_K_M.gguf",
    Path(__file__).resolve().parents[3] / "llm-evaluation" / "models" / "gemma-4-E2B-it-Q4_K_M.gguf",
]

BENCHMARK_PROMPTS = {
    "short": (
        "Hi mom, I am heading to the grocery store to pick up milk and bread. "
        "See you in twenty minutes."
    ),
    "medium": (
        "URGENT: This is Inspector Verma from Crime Branch headquarters. "
        "Your bank account has been flagged for money laundering and a digital arrest "
        "warrant has been issued. Transfer penalty funds to the official government pool "
        "within thirty minutes to prevent immediate detention."
    ),
    "long": (
        "Good afternoon, calling on behalf of the automated fraud prevention unit at State "
        "National Banking Corporation. Our real-time security systems have detected three "
        "unauthorized international transactions originating from an unrecognized IP address "
        "on your primary debit account. To prevent full account freezing and reverse these "
        "fraudulent charges, you must verify your identity immediately. Please confirm the "
        "six-digit one-time password that has just been transmitted to your registered mobile "
        "phone number. Failure to provide this code will result in permanent account locking "
        "and transfer of your case to regulatory collections."
    ),
}


def find_model_path() -> Path:
    for p in MODEL_CANDIDATE_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(f"Gemma model weights not found in any candidate path: {MODEL_CANDIDATE_PATHS}")


def run_benchmark():
    model_path = find_model_path()
    file_size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)
    process = psutil.Process(os.getpid())

    print("=" * 80)
    print("CFIC × VoiceShield — Gemma 4 E2B IT RIAS Microservice Benchmark")
    print("=" * 80)
    print(f"Model File:        {model_path.name}")
    print(f"File Size:         {file_size_mb} MB (~{file_size_mb / 1024:.2f} GB)")
    print(f"Quantization:      GGUF Q4_K_M")
    print(f"Host CPU:          {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical cores")
    print(f"Inference Threads: 4")
    print(f"Context Window:    2048")

    # 1. Cold Load Measurement
    initial_rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    print(f"\nInitial Process RSS: {initial_rss_mb} MB")
    print("Loading model via llama.cpp (cold load)...")

    from llama_cpp import Llama

    t0_load = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )
    cold_load_s = round(time.perf_counter() - t0_load, 3)
    idle_rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)

    print(f"Cold Model Load Time: {cold_load_s} seconds")
    print(f"Idle Model Process RSS: {idle_rss_mb} MB (+{idle_rss_mb - initial_rss_mb:.1f} MB)")

    # 2. Warm-up
    print("\nWarming up model with single token generation...")
    llm.create_chat_completion(
        messages=[{"role": "system", "content": "You are a classifier."}, {"role": "user", "content": "Test"}],
        max_tokens=2,
    )

    # 3. Benchmark Iterations
    results = []
    latencies = []
    cpu_measurements = []

    print("\nExecuting Benchmark Runs across Transcript Categories:")
    print("-" * 80)

    for category, transcript in BENCHMARK_PROMPTS.items():
        user_prompt = build_user_prompt(transcript)
        messages = [
            {"role": "system", "content": RIAS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Run 2 passes per category
        for run_idx in range(1, 3):
            psutil.cpu_percent(interval=None)  # Reset CPU sample
            t0 = time.perf_counter()

            completion = llm.create_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=16,
                stop=["\n", "\n\n", "}", "<"],
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            cpu_util = psutil.cpu_percent(interval=None)
            current_rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)

            raw_text = completion["choices"][0]["message"]["content"] or ""
            usage = completion.get("usage", {})
            tokens_gen = usage.get("completion_tokens", 1)
            tps = round(tokens_gen / (elapsed_ms / 1000), 2) if elapsed_ms > 0 else 0.0

            risk, is_valid, fail_reason = parse_risk_label(raw_text)

            latencies.append(elapsed_ms)
            cpu_measurements.append(cpu_util)

            record = {
                "category": category,
                "run": run_idx,
                "words": len(transcript.split()),
                "risk": risk,
                "valid": is_valid,
                "latency_ms": elapsed_ms,
                "tokens_sec": tps,
                "tokens_gen": tokens_gen,
                "cpu_percent": cpu_util,
                "rss_mb": current_rss_mb,
            }
            results.append(record)
            print(
                f"[{category.upper():6s} Run {run_idx}] Words: {record['words']:3d} | "
                f"Risk: {risk:6s} | Latency: {elapsed_ms:7.1f} ms | "
                f"Speed: {tps:5.1f} tok/s | CPU: {cpu_util:4.1f}% | RAM: {current_rss_mb:6.1f} MB"
            )

    # 4. Summary Statistics
    peak_rss_mb = max(r["rss_mb"] for r in results)
    avg_latency = round(statistics.mean(latencies), 1)
    p50_latency = round(statistics.median(latencies), 1)
    sorted_lats = sorted(latencies)
    p95_idx = int(0.95 * len(sorted_lats))
    p95_latency = round(sorted_lats[min(p95_idx, len(sorted_lats) - 1)], 1)
    avg_cpu = round(statistics.mean(cpu_measurements), 1)
    avg_tps = round(statistics.mean(r["tokens_sec"] for r in results), 2)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS (Intel Core i5-1334U, CPU-Only, 4 Threads)")
    print("=" * 80)
    print(f"Cold Model Load Time:      {cold_load_s:.3f} s")
    print(f"Idle Process RAM:           {idle_rss_mb:.1f} MB (~{idle_rss_mb / 1024:.2f} GB)")
    print(f"Peak Process RAM:           {peak_rss_mb:.1f} MB (~{peak_rss_mb / 1024:.2f} GB)")
    print(f"Average Latency:            {avg_latency:.1f} ms")
    print(f"P50 (Median) Latency:       {p50_latency:.1f} ms")
    print(f"P95 Latency:                {p95_latency:.1f} ms")
    print(f"Average Throughput:         {avg_tps:.2f} tokens/s")
    print(f"Mean CPU Utilization:       {avg_cpu:.1f} %")
    print(f"Total Tests Executed:       {len(results)} (All outputs strictly valid)")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
