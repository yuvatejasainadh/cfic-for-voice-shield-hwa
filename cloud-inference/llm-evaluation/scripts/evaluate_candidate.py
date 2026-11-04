"""Evaluation engine for a single LLM candidate on the RIAS benchmark."""

import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import Any
import psutil
from llama_cpp import Llama

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EVAL_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from prompts.rias_prompt import RIAS_SYSTEM_PROMPT, build_user_prompt, parse_rias_response


def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate overall, per-class, and per-language metrics."""
    classes = ["LOW", "MEDIUM", "HIGH"]
    total = len(results)
    if total == 0:
        return {}

    correct = 0
    invalid_count = 0
    confusion = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    confusion["INCONCLUSIVE"] = {c2: 0 for c2 in classes}

    by_lang: dict[str, list[dict]] = {}
    for r in results:
        lang = r["language"]
        by_lang.setdefault(lang, []).append(r)

        gt = r["ground_truth"]
        pred = r["predicted"]
        is_valid = r["is_valid"]

        if not is_valid or pred not in classes:
            invalid_count += 1
            confusion["INCONCLUSIVE"][gt] += 1
        else:
            confusion[pred][gt] += 1
            if pred == gt:
                correct += 1

    accuracy = correct / total

    # Per-class metrics
    per_class = {}
    macro_p = []
    macro_r = []
    macro_f1 = []

    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[c][actual] for actual in classes if actual != c)
        fn = sum(confusion[other][c] for other in classes if other != c) + confusion["INCONCLUSIVE"][c]

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[k][c] for k in classes) + confusion["INCONCLUSIVE"][c]
        }
        macro_p.append(prec)
        macro_r.append(rec)
        macro_f1.append(f1)

    # Safety metrics
    high_tp = confusion["HIGH"]["HIGH"]
    high_fn = sum(confusion[other]["HIGH"] for other in ["LOW", "MEDIUM"]) + confusion["INCONCLUSIVE"]["HIGH"]
    high_total = high_tp + high_fn
    high_recall = high_tp / high_total if high_total > 0 else 0.0

    # False LOW rate for HIGH ground truth: classified as LOW when actually HIGH
    false_low_for_high = confusion["LOW"]["HIGH"]
    false_low_rate = false_low_for_high / high_total if high_total > 0 else 0.0

    # Per-language accuracy and HIGH recall
    lang_metrics = {}
    for lang, items in by_lang.items():
        lang_total = len(items)
        lang_correct = sum(1 for x in items if x["predicted"] == x["ground_truth"] and x["is_valid"])
        lang_high_actual = [x for x in items if x["ground_truth"] == "HIGH"]
        lang_high_correct = sum(1 for x in lang_high_actual if x["predicted"] == "HIGH" and x["is_valid"])
        lang_metrics[lang] = {
            "total": lang_total,
            "accuracy": round(lang_correct / lang_total, 4) if lang_total > 0 else 0.0,
            "high_recall": round(lang_high_correct / len(lang_high_actual), 4) if lang_high_actual else 1.0,
            "high_samples": len(lang_high_actual)
        }

    return {
        "total_samples": total,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(sum(macro_p) / len(macro_p), 4),
        "macro_recall": round(sum(macro_r) / len(macro_r), 4),
        "macro_f1": round(sum(macro_f1) / len(macro_f1), 4),
        "high_recall": round(high_recall, 4),
        "high_precision": per_class["HIGH"]["precision"],
        "false_low_rate_for_high": round(false_low_rate, 4),
        "invalid_output_rate": round(invalid_count / total, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "per_language": lang_metrics
    }


def evaluate_model(
    model_name: str,
    model_path: Path,
    dataset_path: Path,
    output_dir: Path,
    n_threads: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 64
) -> dict[str, Any]:
    """Run full evaluation benchmark on a candidate model."""
    print(f"\n=======================================================")
    print(f"EVALUATING CANDIDATE: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Dataset path: {dataset_path}")
    print(f"=======================================================")

    proc = psutil.Process(os.getpid())
    ram_before_mb = proc.memory_info().rss / (1024 * 1024)

    # 1. Measure Cold Load Time
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=n_threads,
        verbose=False
    )
    cold_load_sec = round(time.perf_counter() - t0, 3)
    ram_after_load_mb = proc.memory_info().rss / (1024 * 1024)
    idle_ram_mb = round(ram_after_load_mb - ram_before_mb, 2)
    print(f"Cold model load time: {cold_load_sec} s | Idle RAM added: {idle_ram_mb} MB")

    # 2. Warmup & Measure Warm Load (cached recreation)
    # Warmup prompt
    _ = llm("Hello", max_tokens=2, stop=["\n"])
    t0 = time.perf_counter()
    # Lightweight reset simulation
    llm.reset()
    warm_reset_sec = round(time.perf_counter() - t0, 3)

    # 3. Load Dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    latencies_ms = []
    tokens_per_sec_list = []
    cpu_percentages = []
    peak_ram_mb = idle_ram_mb

    print(f"Running inference on {len(dataset)} samples...")

    for i, item in enumerate(dataset):
        transcript = item["transcript"]
        user_prompt = build_user_prompt(transcript)
        messages = [
            {"role": "system", "content": RIAS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # Monitor CPU and RAM
        psutil.cpu_percent(interval=None)  # Reset measurement
        ram_sample_start = proc.memory_info().rss / (1024 * 1024)

        t_start = time.perf_counter()
        completion = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["\n\n", "}"]
        )
        elapsed_sec = time.perf_counter() - t_start
        latency_ms = round(elapsed_sec * 1000, 2)
        latencies_ms.append(latency_ms)

        cpu_used = psutil.cpu_percent(interval=None)
        cpu_percentages.append(cpu_used)

        ram_curr = proc.memory_info().rss / (1024 * 1024)
        if (ram_curr - ram_before_mb) > peak_ram_mb:
            peak_ram_mb = round(ram_curr - ram_before_mb, 2)

        # Token throughput
        usage = completion.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        tps = completion_tokens / elapsed_sec if elapsed_sec > 0 else 0.0
        tokens_per_sec_list.append(tps)

        # Parse output
        raw_text = completion["choices"][0]["message"]["content"] or ""
        # If stop token '}' truncated JSON closing, restore it
        if "{" in raw_text and not raw_text.strip().endswith("}"):
            raw_text = raw_text.strip() + "}"

        pred_risk, reason, is_valid = parse_rias_response(raw_text)

        record = {
            "id": item.get("id", f"sample_{i:04d}"),
            "language": item["language"],
            "category": item.get("category", "unknown"),
            "ground_truth": item["ground_truth"],
            "predicted": pred_risk,
            "reason": reason,
            "is_valid": is_valid,
            "latency_ms": latency_ms,
            "completion_tokens": completion_tokens,
            "tokens_per_sec": round(tps, 2),
            "raw_output": raw_text.replace("\n", " ")
        }
        results.append(record)

        if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
            print(f"  Processed {i + 1}/{len(dataset)} samples | Latest latency: {latency_ms:.1f}ms | Pred: {pred_risk} (GT: {item['ground_truth']})")

    # Latency percentiles
    sorted_lat = sorted(latencies_ms)
    avg_lat = round(sum(sorted_lat) / len(sorted_lat), 2)
    p50_lat = round(sorted_lat[len(sorted_lat) // 2], 2)
    p95_idx = int(len(sorted_lat) * 0.95)
    p95_lat = round(sorted_lat[min(p95_idx, len(sorted_lat) - 1)], 2)
    avg_tps = round(sum(tokens_per_sec_list) / len(tokens_per_sec_list), 2) if tokens_per_sec_list else 0.0
    avg_cpu = round(sum(cpu_percentages) / len(cpu_percentages), 1) if cpu_percentages else 0.0

    # Calculate classification metrics
    metrics = calculate_metrics(results)

    summary = {
        "model_name": model_name,
        "model_path": str(model_path),
        "file_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
        "cold_load_seconds": cold_load_sec,
        "warm_reset_seconds": warm_reset_sec,
        "idle_ram_mb": idle_ram_mb,
        "peak_ram_mb": peak_ram_mb,
        "latency_avg_ms": avg_lat,
        "latency_p50_ms": p50_lat,
        "latency_p95_ms": p95_lat,
        "tokens_per_second": avg_tps,
        "cpu_utilization_pct": avg_cpu,
        "metrics": metrics
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = model_name.lower().replace(".", "_").replace("-", "_")

    json_path = output_dir / f"{slug}_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / f"{slug}_samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "language", "category", "ground_truth", "predicted",
            "is_valid", "latency_ms", "completion_tokens", "tokens_per_sec",
            "reason", "raw_output"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCompleted evaluation for {model_name}!")
    print(f"Results saved to: {json_path}")
    print(f"Sample breakdown: {csv_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | HIGH Recall: {metrics['high_recall']:.4f}")
    print(f"Latency P50: {p50_lat}ms | P95: {p95_lat}ms | Peak RAM: {peak_ram_mb}MB | Avg TPS: {avg_tps}")

    # Explicitly unload model to free RAM
    del llm
    time.sleep(1)
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate_candidate.py <model_name> <model_path> [dataset_path]")
        sys.exit(1)

    m_name = sys.argv[1]
    m_path = Path(sys.argv[2])
    d_path = Path(sys.argv[3]) if len(sys.argv) > 3 else EVAL_DIR / "datasets" / "rias_dataset_test.json"
    out_dir = EVAL_DIR / "results"

    evaluate_model(m_name, m_path, d_path, out_dir)
