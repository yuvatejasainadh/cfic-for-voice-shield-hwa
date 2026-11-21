"""Controlled 24-sample live-container spot check for PellaV2.

Sends 24 balanced audio samples (12 GENUINE, 12 FAKE) across all 4 dataset
categories and 3 acoustic conditions to the live running Docker container at
http://127.0.0.1:8321/predict.
"""

import glob
import json
import os
import random
import sys
import time
import requests

DATASET_ROOT = r"C:\Users\yuvat\Music\Deepfake Dataset"
ENDPOINT_URL = "http://127.0.0.1:8321/predict"

CATEGORIES = [
    ("real_voice_real_bg_wf_4s", "GENUINE"),
    ("real_voice_fake_bg_wf_4s", "GENUINE"),
    ("fake_voice_fake_bg_wf_4s", "FAKE"),
    ("fake_voice_real_bg_wf_4s", "FAKE"),
]

CONDITIONS = ["balanced", "bg_dominant", "voice_dominant"]

def main():
    print(f"Connecting to live PellaV2 container at {ENDPOINT_URL}...")
    try:
        health_resp = requests.get("http://127.0.0.1:8321/health", timeout=5)
        print("Health status:", health_resp.json())
    except Exception as e:
        print(f"Failed to connect to container: {e}")
        sys.exit(1)

    rng = random.Random(2026)
    selected_samples = []

    for cat_dir, ground_truth in CATEGORIES:
        for cond in CONDITIONS:
            folder = os.path.join(DATASET_ROOT, cat_dir, cond)
            files = sorted(glob.glob(os.path.join(folder, "*.wav")))
            chosen = rng.sample(files, 2)
            for f in chosen:
                selected_samples.append({
                    "path": f,
                    "rel_path": os.path.relpath(f, DATASET_ROOT),
                    "filename": os.path.basename(f),
                    "category": cat_dir,
                    "condition": cond,
                    "ground_truth": ground_truth,
                })

    print(f"Total spot-check samples selected: {len(selected_samples)} (12 GENUINE, 12 FAKE)")

    results = []
    tp = fp = tn = fn = 0

    print("\n" + "=" * 105)
    print(f"{'Idx':<4} | {'Condition':<15} | {'Category':<25} | {'GT':<7} | {'Pred':<7} | {'Score':<7} | {'Match?':<6} | {'Latency':<9}")
    print("-" * 105)

    for i, item in enumerate(selected_samples, 1):
        with open(item["path"], "rb") as audio_fp:
            t0 = time.perf_counter()
            resp = requests.post(
                ENDPOINT_URL,
                files={"file": (item["filename"], audio_fp, "audio/wav")},
                timeout=30,
            )
            roundtrip_ms = round((time.perf_counter() - t0) * 1000, 2)

        data = resp.json()
        pred = data.get("prediction", "ERROR")
        score = data.get("score", -1.0)
        gt = item["ground_truth"]
        matched = (pred == gt)

        if gt == "FAKE" and pred == "FAKE":
            tp += 1
        elif gt == "GENUINE" and pred == "FAKE":
            fp += 1
        elif gt == "GENUINE" and pred == "GENUINE":
            tn += 1
        elif gt == "FAKE" and pred == "GENUINE":
            fn += 1

        res_item = {
            "index": i,
            "filename": item["filename"],
            "rel_path": item["rel_path"],
            "category": item["category"],
            "condition": item["condition"],
            "ground_truth": gt,
            "prediction": pred,
            "score": score,
            "matched": matched,
            "container_latency_ms": data.get("latency_ms", 0),
            "roundtrip_ms": roundtrip_ms,
        }
        results.append(res_item)

        print(f"{i:<4} | {item['condition']:<15} | {item['category'][:25]:<25} | {gt:<7} | {pred:<7} | {score:<7.4f} | {str(matched):<6} | {roundtrip_ms}ms")

    print("=" * 105)

    # Metrics
    total = len(results)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    print("\n--- SUMMARY METRICS (24-Sample Live Container Spot Check) ---")
    print(f"True Positives  (TP) : {tp:<2} (Synthetic correctly flagged FAKE)")
    print(f"False Positives (FP) : {fp:<2} (Genuine falsely flagged FAKE)")
    print(f"True Negatives  (TN) : {tn:<2} (Genuine correctly flagged GENUINE)")
    print(f"False Negatives (FN) : {fn:<2} (Synthetic missed as GENUINE)")
    print(f"Accuracy             : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision            : {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall               : {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1-Score             : {f1:.4f}")
    print(f"False Positive Rate  : {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"False Negative Rate  : {fnr:.4f} ({fnr*100:.2f}%)")

    output_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": total,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
        },
        "detailed_results": results,
    }

    out_file = "benchmark/pellav2_live_spotcheck_24_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved detailed results to {out_file}")

if __name__ == "__main__":
    main()
