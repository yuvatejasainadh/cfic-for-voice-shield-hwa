# PellaV2 Acoustic Deepfake Detector Microservice Documentation

## 1. Overview

This document specifies the containerized, internal-only **PellaV2 Acoustic Deepfake Detector** microservice within CloudForge Intelligence Core (CFIC) × VoiceShield:
- **Model Checkpoint Source:** Hugging Face `Sadanie/pellav2-audio-deepfake-detector`
- **Target File:** `weights/pellav2_detector.pt`
- **File Size:** 2,322,248,702 bytes (~2.16 GiB)
- **Hugging Face Commit SHA:** `94b899915ee551fa90d8844961532e4aaab58322`
- **SHA-256 Checksum:** `d21aedfb52e2f87706c2753715a29665482b0f4a034800c31579df5d458559ac`
- **Backbone Architecture:** 300M Parameter Conformer Speech Backbone (`facebook/w2v-bert-2.0` / `Wav2Vec2BertModel`, 24 Conformer layers, 1024 hidden size, 16 attention heads)
- **Pooling & Classification Head:** Learned Softmax Layer Weighting across 25 hidden states, followed by linear projection (1024 -> 1)
- **Quantization:** None (Full FP32 Precision) — replaced previously degraded INT8 quantization
- **Audio Preprocessing Pipeline:**
  1. 16,000 Hz single-channel mono PCM conversion
  2. Mean zero-centering
  3. 20 Hz highpass Butterworth filtering (second order SOS filter)
  4. Duration crop or zero-padding to 4.0 seconds (64,000 samples)
  5. -23.0 LUFS loudness normalization (ITU-R BS.1770 via `pyloudnorm`)
  6. 160-dimensional Kaldi povey log-mel filterbank extraction (80 bins, dither=0, normalized, paired into 160-dim frames)
- **Output Verdict:** `FAKE` (score $\ge 0.50$) or `GENUINE` (score $< 0.50$), where `score = sigmoid(logit)`
- **Execution Target:** CPU-optimized PyTorch execution
- **Container Port:** `8321` (internal private network only)

---

## 2. Microservice Architecture & Data Flow

```text
Incoming Audio Chunk (16kHz PCM WAV)
        │
        ▼
Validation & Bounds Check (File size < 25MB, Valid WAV headers)
        │
        ▼
Audio Preprocessing (20Hz Highpass -> Crop/Pad 4.0s -> -23 LUFS Normalization)
        │
        ▼
Kaldi Povey Filterbank Extraction (160-dim log-mel features)
        │
        ▼
PellaV2 Detector (w2v-bert-2.0 Backbone + Learned Layer Weights + Head)
        │
        ▼
Raw Logit -> Sigmoid Probability Score [0.0, 1.0]
        │
        ▼
FastAPI Endpoint (/predict) -> Returns SpoofPredictionResponse
        │
        ▼
Cloud Inference Orchestrator (concurrent with Vakgyata LID & Whisper/IndicConformer ASR)
        │
        ▼
TCED Decision Engine (API Gateway) -> Synthesizes Acoustic Spoof with RIAS Semantic Risk
```

---

## 3. Environment Variables Reference

| Variable | Default Value | Description |
|---|---|---|
| `PELLAV2_MODEL_PATH` | `weights/pellav2_detector.pt` | Path to the original FP32 PyTorch checkpoint |
| `PELLAV2_CONFIG_PATH` | `config/w2v_bert_config.json` | Path to Wav2Vec2Bert offline model configuration |
| `PELLAV2_CHECKSUM_PATH` | `checksums.txt` | Path to sha256 checksum manifest |
| `PELLAV2_THRESHOLD` | `0.50` | Decision boundary threshold for FAKE vs GENUINE |
| `PELLAV2_PORT` | `8321` | Internal HTTP microservice port |
| `PELLAV2_USE_MOCK` | `false` | Enable deterministic mock responses for isolated unit testing |
| `PELLAV2_DEVICE` | `cpu` | Device target for inference (`cpu`) |
| `PELLAV2_MAX_AUDIO_BYTES` | `26214400` (25 MiB) | Maximum allowed payload size for audio input |

---

## 4. API Endpoints

### `GET /health`
Liveness check returning service status, model version (`pellav2-original`), architecture (`PellaV2-w2v-bert-2.0`), and SHA-256 integrity verification.

### `GET /ready`
Readiness check returning HTTP 200 when weights are loaded in memory and ready for inference, or HTTP 503 when loading or uninitialized.

### `GET /metadata`
Returns architectural metadata, quantization (`None (FP32)`), checksum, model ID (`Sadanie/pellav2-audio-deepfake-detector`), and decision threshold (`0.50`).

### `POST /predict`
Performs acoustic deepfake detection on uploaded audio multipart form-data (`audio/wav`).

**Response Schema:**
```json
{
  "status": "success",
  "prediction": "FAKE",
  "score": 0.9986,
  "threshold": 0.50,
  "model_version": "pellav2-original",
  "latency_ms": 1986.12
}
```

---

## 5. Security & Isolation

- **Non-root Execution:** Runs under unprivileged user `appuser` (UID 10001) in Docker container.
- **Resource Constraints:** Minimum 8GB RAM recommended (memory limit set to 8G in `docker-compose.yml`) to accommodate the 2.32 GB checkpoint and dynamic tensors during loading and inference.
- **Strict Network Isolation:** Deployed inside `cfic-internal-network`, strictly unreachable from the external ingress.
- **Model Deserialization Integrity:** Weights are verified via SHA-256 (`d21aedfb...`) before loading, with zero custom unsafe classes deserialized.
