# PellaV2 K12 INT8 Acoustic Deepfake Detector Microservice Documentation

## 1. Overview

This document specifies the containerized, internal-only **PellaV2 K12 INT8 Acoustic Deepfake Detector** microservice within CloudForge Intelligence Core (CFIC) × VoiceShield:
- **Locked Model Checkpoint:** `varunpadavala/cfic-pellav2-k12-int8`
- **Target File:** `293mb/pellav2_k12_int8_final.pt`
- **File Size:** 292,876,453 bytes (~279.31 MiB)
- **Hugging Face Commit SHA:** `7e07beb19629efd2edd959112d2cebf57cbcfb91`
- **SHA-256 Checksum:** `41c14cd6252fd3e94475b3277eb13b6f8fc463d5edf5b741b521f14bcfda0f98`
- **Architecture:** 12-Layer Conformer (K12) with 1024 hidden size, 16 attention heads, 4096-dim feed-forward, depthwise convolution kernel 31, 160-dim input projection, 13-element layer weighting, and single-scalar linear classification head
- **Quantization:** PyTorch Dynamic INT8 Quantization (`torch.ao.nn.quantized.dynamic.Linear` / `LinearPackedParams`)
- **Calibration Method:** Platt Scaling Sigmoid: `score = 1 / (1 + exp(-(1.5635074958458532 * logit - 0.08237249469081723)))`
- **Input Features:** 160-dimensional log-mel filterbank spectrogram (FFT 1024, hop 320, window 640 at 16,000 Hz)
- **Output Verdict:** `FAKE` (score $\ge 0.50$) or `GENUINE` (score $< 0.50$)
- **Execution Target:** CPU-optimized PyTorch execution (dynamic quantized linear kernels)
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
Feature Extraction (160-dim log-mel filterbank)
        │
        ▼
PellaV2 Conformer K12 (12-layer INT8 dynamically quantized linear layers)
        │
        ▼
Raw Logit -> Platt Sigmoid Calibration (a = 1.5635, b = -0.0824)
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
| `PELLAV2_MODEL_PATH` | `weights/pellav2_k12_int8_final.pt` | Path to the quantized INT8 PyTorch checkpoint |
| `PELLAV2_CALIBRATION_PATH` | `config/k12_calibration.json` | Path to calibration JSON containing Platt scaling constants |
| `PELLAV2_CHECKSUM_PATH` | `checksums.txt` | Path to sha256 checksum manifest |
| `PELLAV2_THRESHOLD` | `0.50` | Decision boundary threshold for FAKE vs GENUINE |
| `PELLAV2_PORT` | `8321` | Internal HTTP microservice port |
| `PELLAV2_USE_MOCK` | `false` | Enable deterministic mock responses for isolated unit testing |
| `PELLAV2_DEVICE` | `cpu` | Device target for inference (`cpu`) |
| `PELLAV2_MAX_AUDIO_BYTES` | `26214400` (25 MiB) | Maximum allowed payload size for audio input |

---

## 4. API Endpoints

### `GET /health`
Liveness check returning service status and loaded model details.

### `GET /ready`
Readiness check returning HTTP 200 when weights are loaded in memory and ready for inference, or HTTP 503 when loading or uninitialized.

### `GET /metadata`
Returns architectural metadata, quantization scheme, checksum, calibration constants, and threshold.

### `POST /predict`
Performs acoustic deepfake detection on uploaded audio multipart form-data (`audio/wav`).

**Response Schema:**
```json
{
  "status": "success",
  "prediction": "FAKE",
  "score": 0.8837,
  "raw_logit": 1.3495,
  "threshold": 0.50,
  "model_version": "pellav2_k12_int8",
  "latency_ms": 1015.42
}
```

---

## 5. Security & Isolation

- **Non-root Execution:** Runs under unprivileged user `appuser` (UID 10001) in Docker container.
- **Resource Constraints:** Constrained to 2GB RAM limit in `docker-compose.yml` to prevent memory exhaustion.
- **Strict Network Isolation:** Deployed inside `cfic-internal-network`, strictly unreachable from the external ingress.
- **Payload Sanitization:** Bounds validation on audio payload size and content MIME types before invoking DSP transformations.
- **Model Deserialization Integrity:** Weights are verified via SHA-256 before loading, with zero custom unsafe classes deserialized.
