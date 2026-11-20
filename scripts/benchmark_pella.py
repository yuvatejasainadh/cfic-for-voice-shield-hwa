"""Empirical benchmark and validation suite for PellaV2 K12 INT8."""

import hashlib
import json
import os
import sys
import time
import numpy as np
import psutil
import soundfile as sf
import torch

# Ensure app path is accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(repo_root, "cloud-inference", "models", "pellav2"))

from app.model import PellaV2Model, extract_160_log_mel_features, load_pellav2_checkpoint
from app.config import settings


def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark():
    print("=" * 60)
    print("CFIC × VoiceShield — PellaV2 K12 INT8 Benchmark & Evaluation")
    print("=" * 60)

    # 1. Checkpoint Verification
    weights_path = settings.model_path
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(repo_root, "cloud-inference", "models", "pellav2", weights_path)

    calib_path = settings.calibration_path
    if not os.path.isabs(calib_path):
        calib_path = os.path.join(repo_root, "cloud-inference", "models", "pellav2", calib_path)

    print(f"\n[1] Checkpoint File Verification:")
    print(f"Path: {weights_path}")
    assert os.path.isfile(weights_path), f"File not found: {weights_path}"
    file_size_bytes = os.path.getsize(weights_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MiB)")

    hasher = hashlib.sha256()
    with open(weights_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest().lower()
    expected_sha = settings.expected_sha256
    print(f"Computed SHA-256: {sha256_hash}")
    print(f"Expected SHA-256: {expected_sha}")
    assert sha256_hash == expected_sha, "Checksum mismatch!"
    print("Checkpoint Integrity: VERIFIED (100% Match)")

    # 2. Cold Loading & Memory Footprint
    print(f"\n[2] Cold Loading & Resource Consumption:")
    mem_before = get_process_memory_mb()
    t0 = time.perf_counter()
    model = load_pellav2_checkpoint(weights_path, calib_path)
    t_load = (time.perf_counter() - t0) * 1000
    mem_after = get_process_memory_mb()
    mem_delta = mem_after - mem_before
    print(f"Cold Model Load Time: {t_load:.2f} ms")
    print(f"Memory RSS Before Load: {mem_before:.2f} MiB")
    print(f"Memory RSS After Load: {mem_after:.2f} MiB")
    print(f"Model Memory Footprint: {mem_delta:.2f} MiB")

    # 3. Warm Latency Benchmark on Test Audio
    test_audio_path = os.path.join(repo_root, "cfic-asr-benchmark", "data", "audio", "telugu", "telugu_00000.wav")
    print(f"\n[3] Warm Inference Latency (10 iterations on real audio):")
    audio, sr = sf.read(test_audio_path)
    duration_s = len(audio) / sr
    print(f"Audio file: {test_audio_path}")
    print(f"Sample Rate: {sr} Hz, Duration: {duration_s:.3f} s, Channels: {audio.ndim}")

    warmup_features = extract_160_log_mel_features(audio, sr)
    with torch.no_grad():
        _ = model(warmup_features)  # Initial warm-up pass

    latencies_feature = []
    latencies_model = []
    latencies_total = []

    for i in range(10):
        t0 = time.perf_counter()
        feats = extract_160_log_mel_features(audio, sr)
        t_feat = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        with torch.no_grad():
            raw_logit, prob = model(feats)
            raw_logit = raw_logit.item()
            prob = prob.item()
        t_mod = (time.perf_counter() - t1) * 1000
        t_tot = (time.perf_counter() - t0) * 1000

        latencies_feature.append(t_feat)
        latencies_model.append(t_mod)
        latencies_total.append(t_tot)

    print(f"Feature Extraction Mean: {np.mean(latencies_feature):.2f} ms (± {np.std(latencies_feature):.2f} ms)")
    print(f"INT8 Model Forward Mean: {np.mean(latencies_model):.2f} ms (± {np.std(latencies_model):.2f} ms)")
    print(f"End-to-End Warm Mean:    {np.mean(latencies_total):.2f} ms (± {np.std(latencies_total):.2f} ms)")
    print(f"Real-Time Factor (RTF):  {np.mean(latencies_total) / (duration_s * 1000):.4f}x")
    print(f"Logit: {raw_logit:.4f}, Calibrated Probability: {prob:.4f}, Prediction: {'FAKE' if prob >= 0.50 else 'GENUINE'}")

    # 4. Input Duration Scaling
    print(f"\n[4] Latency vs Input Audio Duration:")
    durations = [1.0, 3.0, 5.0, 8.0, 10.0]
    for d in durations:
        samples = int(16000 * d)
        synthetic_sig = np.random.randn(samples).astype(np.float32) * 0.1
        f = extract_160_log_mel_features(synthetic_sig, 16000)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(f)
        dt = (time.perf_counter() - t0) * 1000
        rtf = dt / (d * 1000)
        print(f"Duration {d:4.1f}s: Feature shape {list(f.shape)} | Latency {dt:6.2f} ms | RTF {rtf:.4f}x")

    # 5. Ground Truth Evaluation on Benchmark Dataset (XTTS-v2 Synthetic Telugu)
    manifest_path = os.path.join(repo_root, "cfic-asr-benchmark", "data", "manifest_telugu.jsonl")
    print(f"\n[5] Ground Truth Spoof Evaluation (25 Synthetic Samples from manifest_telugu.jsonl):")
    sample_scores = []
    fake_count = 0
    total_evaluated = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 25:
                break
            record = json.loads(line)
            rel_audio = os.path.join(repo_root, "cfic-asr-benchmark", "data", "audio", "telugu", f"{record['sample_id']}.wav")
            if not os.path.exists(rel_audio):
                continue
            sig, sig_sr = sf.read(rel_audio)
            feats = extract_160_log_mel_features(sig, sig_sr)
            with torch.no_grad():
                l, p = model(feats)
                l = l.item()
                score = p.item()
            verdict = "FAKE" if score >= 0.50 else "GENUINE"
            sample_scores.append(score)
            total_evaluated += 1
            if verdict == "FAKE":
                fake_count += 1
            if idx < 5:
                print(f"  Sample {record['sample_id']}: duration={record['audio_duration_seconds']:.2f}s | logit={l:+.4f} | score={score:.4f} | verdict={verdict}")

    recall_spoof = (fake_count / total_evaluated) * 100 if total_evaluated > 0 else 0
    print(f"Evaluated Samples: {total_evaluated}")
    print(f"Classified as FAKE: {fake_count}/{total_evaluated} ({recall_spoof:.1f}%)")
    print(f"Score Mean: {np.mean(sample_scores):.4f} (± {np.std(sample_scores):.4f})")
    print(f"Score Min: {np.min(sample_scores):.4f}, Max: {np.max(sample_scores):.4f}")

    # 6. Evaluation on Baseline Non-Speech / Bona Fide Signals
    print(f"\n[6] Acoustic Discrimination on Non-TTS Baseline Audio:")
    # 440 Hz pure tone (sine)
    t_axis = np.linspace(0, 3.0, int(16000 * 3.0), endpoint=False)
    sine_sig = 0.5 * np.sin(2 * np.pi * 440.0 * t_axis).astype(np.float32)
    sine_feats = extract_160_log_mel_features(sine_sig, 16000)
    with torch.no_grad():
        sine_logit, sine_score = model(sine_feats)
        sine_logit = sine_logit.item()
        sine_score = sine_score.item()
    print(f"  Pure 440Hz Sine Wave: logit={sine_logit:+.4f} | score={sine_score:.4f} | verdict={'FAKE' if sine_score >= 0.50 else 'GENUINE'}")

    # White noise
    noise_sig = np.random.normal(0, 0.05, int(16000 * 3.0)).astype(np.float32)
    noise_feats = extract_160_log_mel_features(noise_sig, 16000)
    with torch.no_grad():
        noise_logit, noise_score = model(noise_feats)
        noise_logit = noise_logit.item()
        noise_score = noise_score.item()
    print(f"  White Noise:          logit={noise_logit:+.4f} | score={noise_score:.4f} | verdict={'FAKE' if noise_score >= 0.50 else 'GENUINE'}")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    benchmark()
