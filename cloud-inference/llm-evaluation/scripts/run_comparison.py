"""Master evaluation runner comparing Candidate A (Qwen3.5-2B) and Candidate B (Gemma-4-E2B-it)."""

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

CANDIDATES = [
    {
        "name": "Qwen3.5-2B",
        "path": MODELS_DIR / "Qwen3.5-2B-Q4_K_M.gguf",
        "repo": "Qwen/Qwen3.5-2B",
        "base_commit": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "gguf_repo": "unsloth/Qwen3.5-2B-GGUF",
        "gguf_commit": "f6d5376be1edb4d416d56da11e5397a961aca8ae",
        "quantization": "Q4_K_M",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "license": "Apache-2.0"
    },
    {
        "name": "Gemma-4-E2B-it",
        "path": MODELS_DIR / "gemma-4-E2B-it-Q4_K_M.gguf",
        "repo": "google/gemma-4-E2B-it",
        "base_commit": "4adf15748de9e1a0097b6274cee9c15836c41c37",
        "gguf_repo": "unsloth/gemma-4-E2B-it-GGUF",
        "gguf_commit": "0314792d7f1f7e229411f620751375812bb9faf2",
        "quantization": "Q4_K_M",
        "architecture": "Gemma4ForConditionalGeneration",
        "license": "Apache-2.0"
    }
]


def run_full_comparison():
    print("=================================================================")
    print("CFIC × VoiceShield — RIAS LLM Candidate Comparative Benchmark")
    print("=================================================================")
    print(f"Dataset: {TEST_DATASET_PATH}")
    print(f"Results Directory: {RESULTS_DIR}")

    comparison_results = {}

    for cand in CANDIDATES:
        c_name = cand["name"]
        c_path = cand["path"]

        if not c_path.exists():
            raise FileNotFoundError(f"Missing model file: {c_path}")

        # 1. Classification & Hardware Benchmark (Sequentially)
        bench_summary = evaluate_model(
            model_name=c_name,
            model_path=c_path,
            dataset_path=TEST_DATASET_PATH,
            output_dir=RESULTS_DIR,
            n_threads=4,
            temperature=0.0,
            max_tokens=64
        )

        # Allow cooling / system settling
        time.sleep(2)

        # 2. Robustness Suite
        rob_summary = run_robustness_tests(
            model_name=c_name,
            model_path=c_path
        )

        comparison_results[c_name] = {
            "metadata": cand,
            "benchmark": bench_summary,
            "robustness": rob_summary
        }

        # Settle before next candidate
        time.sleep(3)

    # Save consolidated comparison JSON
    out_comparison_path = RESULTS_DIR / "comparison_summary.json"
    with open(out_comparison_path, "w", encoding="utf-8") as f:
        # Convert Path objects to string
        clean_json = json.loads(json.dumps(comparison_results, default=str))
        json.dump(clean_json, f, indent=2, ensure_ascii=False)

    print("\n=================================================================")
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Consolidated results saved to: {out_comparison_path}")
    print("=================================================================")


if __name__ == "__main__":
    run_full_comparison()
