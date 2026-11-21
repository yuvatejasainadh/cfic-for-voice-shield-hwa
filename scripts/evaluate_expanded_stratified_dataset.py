"""Expanded Stratified Benchmark Evaluation of Original PellaV2 on Deepfake Audio Dataset.

Samples 20 per subcategory across all 12 acoustic subsets (240 total samples: 120 FAKE, 120 GENUINE).
"""

import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List

import httpx
import numpy as np

# Ensure api-gateway is on path for TCED
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api-gateway"))
from app.orchestration.decision_engine import DecisionEngine

DATASET_ROOT = Path(r"C:\Users\yuvat\Music\Deepfake Dataset")
PELLAV2_URL = "http://127.0.0.1:8321"
SAMPLES_PER_SUBCATEGORY = 20  # 12 subcategories * 20 = 240 balanced samples
RANDOM_SEED = 123


def compute_roc_auc(y_true: List[int], y_scores: List[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_scores))
    except ImportError:
        pairs = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
        pos = sum(y_true)
        neg = len(y_true) - pos
        if pos == 0 or neg == 0:
            return 0.5
        tp = 0
        fp = 0
        prev_tp = 0
        prev_fp = 0
        auc = 0.0
        for score, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
                auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
                prev_tp = tp
                prev_fp = fp
        return auc / (pos * neg)


def main():
    print("=" * 80)
    print("EXPANDED STRATIFIED EVALUATION (240 Samples) - ORIGINAL PELLAV2")
    print("=" * 80)

    random.seed(RANDOM_SEED)
    top_level_dirs = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir()])
    
    samples = []
    for cat_dir in top_level_dirs:
        gt_label = "FAKE" if "fake_voice" in cat_dir.name else "GENUINE"
        subdirs = sorted([s for s in cat_dir.iterdir() if s.is_dir()])
        for sub in subdirs:
            wav_files = sorted(list(sub.glob("*.wav")))
            selected = random.sample(wav_files, min(SAMPLES_PER_SUBCATEGORY, len(wav_files)))
            for f in selected:
                samples.append({
                    "path": f,
                    "filename": f.name,
                    "category": cat_dir.name,
                    "condition": sub.name,
                    "ground_truth": gt_label,
                })

    total = len(samples)
    print(f"Sampled {total} audio clips (120 FAKE, 120 GENUINE)")

    predictions = []
    latencies = []
    engine = DecisionEngine(calibrated=True)
    client = httpx.Client(timeout=60.0)

    for idx, item in enumerate(samples):
        with open(item["path"], "rb") as af:
            content = af.read()

        t0 = time.perf_counter()
        resp = client.post(
            f"{PELLAV2_URL}/predict",
            files={"file": (item["filename"], content, "audio/wav")},
        )
        lat_ms = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(lat_ms)

        if resp.status_code != 200:
            print(f"Failed {item['filename']}: {resp.status_code}")
            continue

        d = resp.json()
        pred = d.get("prediction")
        score = float(d.get("score", 0.0))

        tced_out, tced_reason = engine.decide(
            {"status": "success", "prediction": pred, "score": score},
            {"status": "success", "risk_level": "LOW"},
        )

        record = {
            "filename": item["filename"],
            "category": item["category"],
            "condition": item["condition"],
            "ground_truth": item["ground_truth"],
            "predicted_label": pred,
            "spoof_score": score,
            "is_correct": (pred == item["ground_truth"]),
            "latency_ms": lat_ms,
            "tced_outcome": tced_out.value,
            "tced_reason": tced_reason,
        }
        predictions.append(record)

        if (idx + 1) % 20 == 0 or (idx + 1) == total:
            print(f"Processed {idx + 1}/{total} (current avg latency: {np.mean(latencies):.2f} ms)...", flush=True)

    client.close()

    tp = sum(1 for p in predictions if p["ground_truth"] == "FAKE" and p["predicted_label"] == "FAKE")
    fp = sum(1 for p in predictions if p["ground_truth"] == "GENUINE" and p["predicted_label"] == "FAKE")
    tn = sum(1 for p in predictions if p["ground_truth"] == "GENUINE" and p["predicted_label"] == "GENUINE")
    fn = sum(1 for p in predictions if p["ground_truth"] == "FAKE" and p["predicted_label"] == "GENUINE")

    acc = (tp + tn) / len(predictions)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    y_true = [1 if p["ground_truth"] == "FAKE" else 0 for p in predictions]
    y_scores = [p["spoof_score"] for p in predictions]
    roc_auc = compute_roc_auc(y_true, y_scores)

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    p95_lat = float(np.percentile(lat_arr, 95))
    rtf = (mean_lat / 1000.0) / 4.0

    # Category breakdown
    cat_metrics = {}
    for cat in sorted(list(set(p["category"] for p in predictions))):
        cp = [p for p in predictions if p["category"] == cat]
        c_tp = sum(1 for p in cp if p["ground_truth"] == "FAKE" and p["predicted_label"] == "FAKE")
        c_fp = sum(1 for p in cp if p["ground_truth"] == "GENUINE" and p["predicted_label"] == "FAKE")
        c_tn = sum(1 for p in cp if p["ground_truth"] == "GENUINE" and p["predicted_label"] == "GENUINE")
        c_fn = sum(1 for p in cp if p["ground_truth"] == "FAKE" and p["predicted_label"] == "GENUINE")
        c_acc = (c_tp + c_tn) / len(cp)
        cat_metrics[cat] = {
            "total": len(cp), "accuracy": round(c_acc, 4),
            "tp": c_tp, "fp": c_fp, "tn": c_tn, "fn": c_fn,
        }

    # Condition breakdown
    cond_metrics = {}
    for cond in sorted(list(set(p["condition"] for p in predictions))):
        cp = [p for p in predictions if p["condition"] == cond]
        c_tp = sum(1 for p in cp if p["ground_truth"] == "FAKE" and p["predicted_label"] == "FAKE")
        c_fp = sum(1 for p in cp if p["ground_truth"] == "GENUINE" and p["predicted_label"] == "FAKE")
        c_tn = sum(1 for p in cp if p["ground_truth"] == "GENUINE" and p["predicted_label"] == "GENUINE")
        c_fn = sum(1 for p in cp if p["ground_truth"] == "FAKE" and p["predicted_label"] == "GENUINE")
        c_acc = (c_tp + c_tn) / len(cp)
        cond_metrics[cond] = {
            "total": len(cp), "accuracy": round(c_acc, 4),
            "tp": c_tp, "fp": c_fp, "tn": c_tn, "fn": c_fn,
        }

    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_evaluated": len(predictions),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "latency_metrics": {"mean_ms": round(mean_lat, 2), "p95_ms": round(p95_lat, 2), "rtf": round(rtf, 4)},
        "category_metrics": cat_metrics,
        "condition_metrics": cond_metrics,
    }

    out_json = Path("benchmark/pellav2_expanded_240_evaluation_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- EXPANDED BENCHMARK RESULTS (240 SAMPLES) ---")
    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"Precision:       {prec * 100:.2f}%")
    print(f"Recall:          {rec * 100:.2f}%")
    print(f"F1-Score:        {f1:.4f}")
    print(f"ROC-AUC:         {roc_auc:.4f}")
    print(f"FPR:             {fpr * 100:.2f}% (FP={fp}, TN={tn})")
    print(f"FNR:             {fnr * 100:.2f}% (FN={fn}, TP={tp})")
    print(f"Mean Latency:    {mean_lat:.2f} ms (P95: {p95_lat:.2f} ms, RTF: {rtf:.4f})")
    print("Categories:")
    for cat, m in cat_metrics.items():
        print(f"  {cat:28s}: Acc={m['accuracy']*100:6.2f}% (TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']})")
    print("Conditions:")
    for cond, m in cond_metrics.items():
        print(f"  {cond:16s}: Acc={m['accuracy']*100:6.2f}% (TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']})")


if __name__ == "__main__":
    main()
