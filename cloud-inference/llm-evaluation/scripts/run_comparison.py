"""Evaluation runner for the selected RIAS LLM (Gemma-4-E2B-it).

Note: Historical candidate comparison data between Qwen3.5-2B (rejected) and Gemma 4 E2B IT
(selected) is archived in docs/architecture/llm-model-evaluation-report.md and
cloud-inference/llm-evaluation/results/comparison_summary.json for decision provenance.
Active runtime execution is strictly locked to Gemma-4-E2B-it.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EVAL_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from scripts.evaluate_candidate import evaluate_model
from tests.test_robustness import run_robustness_tests

MODELS_DIR = EVAL_DIR / "models"
DATASETS_DIR = EVAL_DIR / "datasets"
RESULTS_DIR = EVAL_DIR / "results"
TEST_DATASET_PATH = DATASETS_DIR / "rias_dataset_test.json"

SELECTED_MODEL = {
    "name": "Gemma-4-E2B-it",
    "status": "LOCKED_RIAS_LLM",
    "path": MODELS_DIR / "gemma-4-E2B-it-Q4_K_M.gguf",
    "repo": "google/gemma-4-E2B-it",
    "base_commit": "4adf15748de9e1a0097b6274cee9c15836c41c37",
    "gguf_repo": "unsloth/gemma-4-E2B-it-GGUF",
    "gguf_commit": "0314792d7f1f7e229411f620751375812bb9faf2",
    "quantization": "Q4_K_M",
    "architecture": "Gemma4ForConditionalGeneration",
    "license": "Apache-2.0"
}


def run_evaluation():
    print("=================================================================")
    print("CFIC × VoiceShield — Locked RIAS LLM Benchmark (Gemma-4-E2B-it)")
    print("=================================================================")
    print(f"Dataset: {TEST_DATASET_PATH}")
    print(f"Results Directory: {RESULTS_DIR}")

    m_name = SELECTED_MODEL["name"]
    m_path = SELECTED_MODEL["path"]

    if not m_path.exists():
        raise FileNotFoundError(f"Missing locked model file: {m_path}")

    # 1. Classification & Hardware Benchmark
    bench_summary = evaluate_model(
        model_name=m_name,
        model_path=m_path,
        dataset_path=TEST_DATASET_PATH,
        output_dir=RESULTS_DIR,
        n_threads=4,
        temperature=0.0,
        max_tokens=64
    )

    time.sleep(2)

    # 2. Robustness Suite
    rob_summary = run_robustness_tests(
        model_name=m_name,
        model_path=m_path
    )

    print("\n=================================================================")
    print("BENCHMARK COMPLETED SUCCESSFULLY FOR GEMMA-4-E2B-IT!")
    print("=================================================================")


if __name__ == "__main__":
    run_evaluation()
