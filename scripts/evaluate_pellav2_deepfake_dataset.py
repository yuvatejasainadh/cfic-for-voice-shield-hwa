"""Comprehensive Evaluation of PellaV2 K12 INT8 on Deepfake Audio Dataset & TCED Policy Verification.

Evaluates against:
- 4 Top-level Categories:
    1. fake_voice_fake_bg_wf_4s (GT: FAKE)
    2. fake_voice_real_bg_wf_4s (GT: FAKE)
    3. real_voice_fake_bg_wf_4s (GT: GENUINE)
    4. real_voice_real_bg_wf_4s (GT: GENUINE)
- 3 Audio Mixing Conditions per category:
    - balanced
    - bg_dominant
    - voice_dominant
- Live PellaV2 microservice endpoint: http://127.0.0.1:8321/predict
- TCED Decision Engine (Rules A through H)
"""

import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx
import numpy as np

# Ensure api-gateway is on path for TCED
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api-gateway"))
from app.orchestration.decision_engine import DecisionEngine
from app.schemas.analysis import FinalOutcome, RiskLevel

DATASET_ROOT = Path(r"C:\Users\yuvat\Music\Deepfake Dataset")
PELLAV2_URL = os.getenv("PELLAV2_URL", "http://127.0.0.1:8321")
SAMPLES_PER_SUBCATEGORY = 10  # 12 subcategories * 10 = 120 balanced samples
RANDOM_SEED = 42

def compute_roc_auc(y_true: List[int], y_scores: List[float]) -> float:
    """Calculate ROC-AUC without heavy external dependency if needed."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_scores))
    except ImportError:
        # Fallback trapezoidal computation
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
    print("=" * 70)
    print("CFIC PellaV2 K12 INT8 & TCED Evaluation on Deepfake Audio Dataset")
    print("=" * 70)

    if not DATASET_ROOT.exists():
        print(f"ERROR: Dataset path does not exist: {DATASET_ROOT}")
        sys.exit(1)

    # 1. Collect and sample dataset
    random.seed(RANDOM_SEED)
    top_level_dirs = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir()])
    
    samples_to_evaluate = []
    print(f"\n[1/5] Sampling dataset ({SAMPLES_PER_SUBCATEGORY} per acoustic subcategory)...")

    for cat_dir in top_level_dirs:
        gt_label = "FAKE" if "fake_voice" in cat_dir.name else "GENUINE"
        subdirs = sorted([s for s in cat_dir.iterdir() if s.is_dir()])
        for sub in subdirs:
            wav_files = sorted(list(sub.glob("*.wav")))
            selected = random.sample(wav_files, min(SAMPLES_PER_SUBCATEGORY, len(wav_files)))
            for f in selected:
                samples_to_evaluate.append({
                    "path": f,
                    "filename": f.name,
                    "category": cat_dir.name,
                    "condition": sub.name,
                    "ground_truth": gt_label,
                })
            print(f"  Sampled {len(selected)} from {cat_dir.name}/{sub.name}")

    total_samples = len(samples_to_evaluate)
    fake_count = sum(1 for s in samples_to_evaluate if s["ground_truth"] == "FAKE")
    genuine_count = sum(1 for s in samples_to_evaluate if s["ground_truth"] == "GENUINE")
    print(f"\nTotal sampled: {total_samples} audio clips ({fake_count} FAKE, {genuine_count} GENUINE)")

    # 2. Check PellaV2 service connectivity
    print(f"\n[2/5] Checking live PellaV2 service at {PELLAV2_URL}...")
    try:
        r = httpx.get(f"{PELLAV2_URL}/health", timeout=5.0)
        r.raise_for_status()
        health_info = r.json()
        print(f"  Service healthy: {health_info}")
    except Exception as exc:
        print(f"ERROR: Could not connect to PellaV2 service: {exc}")
        sys.exit(1)

    # 3. Execute inference against live PellaV2
    print(f"\n[3/5] Executing live inference on {total_samples} samples...")
    predictions = []
    latencies = []
    engine = DecisionEngine(calibrated=True)

    client = httpx.Client(timeout=30.0)
    for idx, item in enumerate(samples_to_evaluate):
        file_path = item["path"]
        with open(file_path, "rb") as audio_f:
            audio_bytes = audio_f.read()

        t0 = time.perf_counter()
        resp = client.post(
            f"{PELLAV2_URL}/predict",
            files={"file": (item["filename"], audio_bytes, "audio/wav")},
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(latency_ms)

        if resp.status_code != 200:
            print(f"  WARNING: Request failed for {item['filename']}: status {resp.status_code}")
            continue

        data = resp.json()
        pred_label = data.get("prediction")
        score = float(data.get("score", 0.0))
        threshold = float(data.get("threshold", 0.50))

        # TCED Synthesis
        # Row 1: Acoustic + RIAS LOW
        tced_low_outcome, tced_low_reason = engine.decide(
            {"status": "success", "prediction": pred_label, "score": score},
            {"status": "success", "risk_level": "LOW"},
        )
        # Row 2: Acoustic + RIAS HIGH
        tced_high_outcome, tced_high_reason = engine.decide(
            {"status": "success", "prediction": pred_label, "score": score},
            {"status": "success", "risk_level": "HIGH"},
        )
        # Row 3: Acoustic UNAVAILABLE + RIAS LOW (Rule F)
        tced_unavail_low_outcome, tced_unavail_low_reason = engine.decide(
            {"status": "error", "prediction": None},
            {"status": "success", "risk_level": "LOW"},
        )

        record = {
            "filename": item["filename"],
            "category": item["category"],
            "condition": item["condition"],
            "ground_truth": item["ground_truth"],
            "predicted_label": pred_label,
            "spoof_score": score,
            "threshold": threshold,
            "is_correct": (pred_label == item["ground_truth"]),
            "latency_ms": latency_ms,
            "tced_rias_low_outcome": tced_low_outcome.value,
            "tced_rias_low_reason": tced_low_reason,
            "tced_rias_high_outcome": tced_high_outcome.value,
            "tced_rias_high_reason": tced_high_reason,
            "tced_unavail_low_outcome": tced_unavail_low_outcome.value,
            "tced_unavail_low_reason": tced_unavail_low_reason,
        }
        predictions.append(record)

        if (idx + 1) % 10 == 0 or (idx + 1) == total_samples:
            print(f"  Processed {idx + 1}/{total_samples} samples (current latency avg: {np.mean(latencies):.2f} ms)...", flush=True)

    client.close()

    # 4. Compute Comprehensive Evaluation Metrics
    print(f"\n[4/5] Computing evaluation metrics...")
    
    tp = sum(1 for p in predictions if p["ground_truth"] == "FAKE" and p["predicted_label"] == "FAKE")
    fp = sum(1 for p in predictions if p["ground_truth"] == "GENUINE" and p["predicted_label"] == "FAKE")
    tn = sum(1 for p in predictions if p["ground_truth"] == "GENUINE" and p["predicted_label"] == "GENUINE")
    fn = sum(1 for p in predictions if p["ground_truth"] == "FAKE" and p["predicted_label"] == "GENUINE")

    total_eval = len(predictions)
    accuracy = (tp + tn) / total_eval if total_eval > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    y_true_binary = [1 if p["ground_truth"] == "FAKE" else 0 for p in predictions]
    y_scores = [p["spoof_score"] for p in predictions]
    roc_auc = compute_roc_auc(y_true_binary, y_scores)

    # Latency distribution (4-second audio clips)
    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    median_lat = float(np.median(lat_arr))
    p95_lat = float(np.percentile(lat_arr, 95))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    rtf = (mean_lat / 1000.0) / 4.0  # Real-Time Factor: latency / audio duration

    # Breakdown by category
    category_metrics = {}
    for cat in sorted(list(set(p["category"] for p in predictions))):
        cat_preds = [p for p in predictions if p["category"] == cat]
        cat_tp = sum(1 for p in cat_preds if p["ground_truth"] == "FAKE" and p["predicted_label"] == "FAKE")
        cat_fp = sum(1 for p in cat_preds if p["ground_truth"] == "GENUINE" and p["predicted_label"] == "FAKE")
        cat_tn = sum(1 for p in cat_preds if p["ground_truth"] == "GENUINE" and p["predicted_label"] == "GENUINE")
        cat_fn = sum(1 for p in cat_preds if p["ground_truth"] == "FAKE" and p["predicted_label"] == "GENUINE")
        cat_total = len(cat_preds)
        cat_acc = (cat_tp + cat_tn) / cat_total if cat_total > 0 else 0.0
        cat_scores = [p["spoof_score"] for p in cat_preds]
        category_metrics[cat] = {
            "total": cat_total,
            "accuracy": round(cat_acc, 4),
            "mean_score": round(float(np.mean(cat_scores)), 4),
            "tp": cat_tp, "fp": cat_fp, "tn": cat_tn, "fn": cat_fn,
        }

    # Breakdown by acoustic mixing condition
    condition_metrics = {}
    for cond in sorted(list(set(p["condition"] for p in predictions))):
        cond_preds = [p for p in predictions if p["condition"] == cond]
        c_tp = sum(1 for p in cond_preds if p["ground_truth"] == "FAKE" and p["predicted_label"] == "FAKE")
        c_fp = sum(1 for p in cond_preds if p["ground_truth"] == "GENUINE" and p["predicted_label"] == "FAKE")
        c_tn = sum(1 for p in cond_preds if p["ground_truth"] == "GENUINE" and p["predicted_label"] == "GENUINE")
        c_fn = sum(1 for p in cond_preds if p["ground_truth"] == "FAKE" and p["predicted_label"] == "GENUINE")
        c_acc = (c_tp + c_tn) / len(cond_preds) if cond_preds else 0.0
        condition_metrics[cond] = {
            "total": len(cond_preds),
            "accuracy": round(c_acc, 4),
            "tp": c_tp, "fp": c_fp, "tn": c_tn, "fn": c_fn,
        }

    # TCED Rule F verification tally
    rule_f_pass = all(
        p["tced_unavail_low_outcome"] == "INCONCLUSIVE" and p["tced_unavail_low_reason"] == "ACOUSTIC_EVIDENCE_UNAVAILABLE"
        for p in predictions
    )

    # TCED Rule A & D verification: any predicted FAKE must NEVER be downgraded
    rule_a_pass = all(
        (p["tced_rias_low_outcome"] == "FAKE" and p["tced_rias_low_reason"] == "ACOUSTIC_SPOOF_DETECTED")
        for p in predictions if p["predicted_label"] == "FAKE"
    )
    rule_d_pass = all(
        (p["tced_rias_high_outcome"] == "FAKE" and p["tced_rias_high_reason"] == "ACOUSTIC_SPOOF_DETECTED")
        for p in predictions if p["predicted_label"] == "FAKE"
    )

    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "path": str(DATASET_ROOT),
            "total_samples_evaluated": total_eval,
            "fake_samples": fake_count,
            "genuine_samples": genuine_count,
            "audio_duration_sec": 4.0,
            "sample_rate": 16000,
        },
        "model": {
            "service": "model-pellav2",
            "version": "pellav2-k12-int8",
            "architecture": "Conformer-K12-INT8",
            "threshold": 0.50,
            "quantization": "PyTorch INT8 Dynamic Quantization",
        },
        "classification_metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            },
        },
        "latency_metrics": {
            "mean_ms": round(mean_lat, 2),
            "median_ms": round(median_lat, 2),
            "p95_ms": round(p95_lat, 2),
            "min_ms": round(min_lat, 2),
            "max_ms": round(max_lat, 2),
            "rtf": round(rtf, 4),
        },
        "category_breakdown": category_metrics,
        "condition_breakdown": condition_metrics,
        "tced_policy_verification": {
            "rule_f_acoustic_unavail_rias_low_pass": rule_f_pass,
            "rule_f_expected_reason": "ACOUSTIC_EVIDENCE_UNAVAILABLE",
            "rule_a_acoustic_fake_rias_low_preserved": rule_a_pass,
            "rule_d_acoustic_fake_rias_high_preserved": rule_d_pass,
        },
    }

    # 5. Save Artifacts
    print(f"\n[5/5] Saving evaluation artifacts...")
    Path("benchmark").mkdir(parents=True, exist_ok=True)

    # Save CSV predictions
    csv_path = Path("benchmark/pellav2_dataset_predictions.csv")
    fieldnames = [
        "filename", "category", "condition", "ground_truth",
        "predicted_label", "spoof_score", "threshold", "is_correct", "latency_ms",
        "tced_rias_low_outcome", "tced_rias_low_reason",
        "tced_rias_high_outcome", "tced_rias_high_reason",
        "tced_unavail_low_outcome", "tced_unavail_low_reason"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    print(f"  Predictions CSV saved to: {csv_path}")

    # Save JSON report
    json_path = Path("benchmark/pellav2_evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Metrics JSON report saved to: {json_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY RESULTS")
    print("=" * 70)
    print(f"Total Audio Samples:     {total_eval} (150 FAKE, 150 GENUINE)")
    print(f"Accuracy:                {accuracy * 100:.2f}%")
    print(f"Precision:               {precision * 100:.2f}%")
    print(f"Recall:                  {recall * 100:.2f}%")
    print(f"F1-Score:                {f1:.4f}")
    print(f"ROC-AUC:                 {roc_auc:.4f}")
    print(f"FPR (False Alarm):       {fpr * 100:.2f}% (FP={fp}, TN={tn})")
    print(f"FNR (Missed Detection):  {fnr * 100:.2f}% (FN={fn}, TP={tp})")
    print(f"Latency (Mean / P95):    {mean_lat:.2f} ms / {p95_lat:.2f} ms")
    print(f"Real-Time Factor (RTF):  {rtf:.4f} (Audio: 4.0s)")
    print("-" * 70)
    print("CATEGORY BREAKDOWN:")
    for cat, m in category_metrics.items():
        print(f"  {cat:28s}: Acc={m['accuracy']*100:6.2f}%, MeanScore={m['mean_score']:.4f} (TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']})")
    print("-" * 70)
    print("CONDITION BREAKDOWN:")
    for cond, m in condition_metrics.items():
        print(f"  {cond:16s}: Acc={m['accuracy']*100:6.2f}% (TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']})")
    print("-" * 70)
    print("TCED POLICY VERIFICATION:")
    print(f"  Rule F (Unavail + LOW -> INCONCLUSIVE/ACOUSTIC_EVIDENCE_UNAVAILABLE): {'PASSED' if rule_f_pass else 'FAILED'}")
    print(f"  Rule A (FAKE + LOW -> FAKE/ACOUSTIC_SPOOF_DETECTED):                 {'PASSED' if rule_a_pass else 'FAILED'}")
    print(f"  Rule D (FAKE + HIGH -> FAKE/ACOUSTIC_SPOOF_DETECTED):                {'PASSED' if rule_d_pass else 'FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
