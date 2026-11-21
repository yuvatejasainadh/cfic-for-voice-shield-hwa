"""Independent Verification and Audit of Benchmark Reproduction."""

import csv
import json
import time
from pathlib import Path
from typing import List

import httpx
import numpy as np

DATASET_ROOT = Path(r"C:\Users\yuvat\Music\Deepfake Dataset")
PREDICTIONS_CSV = Path("benchmark/pellav2_original_dataset_predictions.csv")
REPORT_JSON = Path("benchmark/pellav2_original_evaluation_report.json")
PELLAV2_URL = "http://127.0.0.1:8321"


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
    print("INDEPENDENT BENCHMARK REPRODUCTION AUDIT")
    print("=" * 80)

    if not PREDICTIONS_CSV.exists():
        print(f"ERROR: {PREDICTIONS_CSV} not found")
        return

    with open(PREDICTIONS_CSV, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    print(f"Total samples recorded: {len(records)}")

    # 1. Validate files on disk
    missing = 0
    for r in records:
        p = DATASET_ROOT / r["category"] / r["condition"] / r["filename"]
        if not p.exists():
            missing += 1
    print(f"Disk file integrity: {len(records) - missing}/{len(records)} found on disk (missing: {missing})")

    # 2. Independent Metric Calculation
    tp = sum(1 for r in records if r["ground_truth"] == "FAKE" and r["predicted_label"] == "FAKE")
    fp = sum(1 for r in records if r["ground_truth"] == "GENUINE" and r["predicted_label"] == "FAKE")
    tn = sum(1 for r in records if r["ground_truth"] == "GENUINE" and r["predicted_label"] == "GENUINE")
    fn = sum(1 for r in records if r["ground_truth"] == "FAKE" and r["predicted_label"] == "GENUINE")

    total = len(records)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    y_true = [1 if r["ground_truth"] == "FAKE" else 0 for r in records]
    y_scores = [float(r["spoof_score"]) for r in records]
    roc_auc = compute_roc_auc(y_true, y_scores)

    latencies = [float(r["latency_ms"]) for r in records]
    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    p50_lat = float(np.median(lat_arr))
    p95_lat = float(np.percentile(lat_arr, 95))
    p99_lat = float(np.percentile(lat_arr, 99))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    rtf = (mean_lat / 1000.0) / 4.0

    print("\n--- INDEPENDENTLY RECALCULATED METRICS ---")
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Negatives (FN): {fn}")
    print(f"Accuracy:             {acc * 100:.2f}% ({acc:.4f})")
    print(f"Precision:            {prec * 100:.2f}% ({prec:.4f})")
    print(f"Recall:               {rec * 100:.2f}% ({rec:.4f})")
    print(f"F1-Score:             {f1:.4f}")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"FPR (False Alarm):    {fpr * 100:.2f}% ({fpr:.4f})")
    print(f"FNR (Missed Spoof):   {fnr * 100:.2f}% ({fnr:.4f})")
    print(f"Mean Latency:         {mean_lat:.2f} ms")
    print(f"P50 (Median) Latency: {p50_lat:.2f} ms")
    print(f"P95 Latency:          {p95_lat:.2f} ms")
    print(f"P99 Latency:          {p99_lat:.2f} ms")
    print(f"Min / Max Latency:    {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"Real-Time Factor:     {rtf:.4f} (Audio: 4.0s)")

    # 3. Check concordance with JSON report
    if REPORT_JSON.exists():
        with open(REPORT_JSON, "r", encoding="utf-8") as f:
            rep = json.load(f)
        rep_cm = rep["classification_metrics"]
        print("\n--- REPORT CONCORDANCE CHECK ---")
        print(f"Report Accuracy:  {rep_cm['accuracy']*100:.2f}% vs Recalculated: {acc*100:.2f}% -> Match: {abs(rep_cm['accuracy'] - acc) < 1e-4}")
        print(f"Report Precision: {rep_cm['precision']*100:.2f}% vs Recalculated: {prec*100:.2f}% -> Match: {abs(rep_cm['precision'] - prec) < 1e-4}")
        print(f"Report Recall:    {rep_cm['recall']*100:.2f}% vs Recalculated: {rec*100:.2f}% -> Match: {abs(rep_cm['recall'] - rec) < 1e-4}")
        print(f"Report F1:        {rep_cm['f1_score']:.4f} vs Recalculated: {f1:.4f} -> Match: {abs(rep_cm['f1_score'] - f1) < 1e-4}")
        print(f"Report ROC-AUC:   {rep_cm['roc_auc']:.4f} vs Recalculated: {roc_auc:.4f} -> Match: {abs(rep_cm['roc_auc'] - roc_auc) < 1e-4}")
        print(f"Report FPR:       {rep_cm['false_positive_rate']*100:.2f}% vs Recalculated: {fpr*100:.2f}% -> Match: {abs(rep_cm['false_positive_rate'] - fpr) < 1e-4}")

    # 4. Live Spot-Check of 6 samples against http://127.0.0.1:8321/predict
    print("\n--- LIVE INFERENCE SPOT-CHECK (6 Samples) ---")
    spot_indices = [0, 20, 40, 60, 80, 100]
    client = httpx.Client(timeout=30.0)
    for idx in spot_indices:
        r = records[idx]
        file_path = DATASET_ROOT / r["category"] / r["condition"] / r["filename"]
        with open(file_path, "rb") as af:
            content = af.read()
        resp = client.post(f"{PELLAV2_URL}/predict", files={"file": (r["filename"], content, "audio/wav")})
        if resp.status_code == 200:
            d = resp.json()
            live_score = float(d["score"])
            live_pred = d["prediction"]
            csv_score = float(r["spoof_score"])
            diff = abs(live_score - csv_score)
            print(f"Sample {idx:3d} [{r['category'][:15]}] GT={r['ground_truth']:7s} | Live: {live_pred:7s} ({live_score:.4f}) | CSV: {r['predicted_label']:7s} ({csv_score:.4f}) | Diff: {diff:.4f} -> {'EXACT' if diff < 0.005 else 'MISMATCH'}")
        else:
            print(f"Sample {idx} failed with status {resp.status_code}")
    client.close()


if __name__ == "__main__":
    main()
