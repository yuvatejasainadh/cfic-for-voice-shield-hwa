# Gemma 4 E2B IT RIAS LLM Microservice Documentation

## 1. Overview

This document specifies the containerized, internal-only **Risk Identification and Assessment Subsystem (RIAS)** LLM microservice within CloudForge Intelligence Core (CFIC) × VoiceShield:
- **Locked Model Checkpoint:** `google/gemma-4-E2B-it`
- **GGUF Repository:** `unsloth/gemma-4-E2B-it-GGUF`
- **GGUF Commit SHA:** `0314792d7f1f7e229411f620751375812bb9faf2`
- **Base Model Commit SHA:** `4adf15748de9e1a0097b6274cee9c15836c41c37`
- **Quantization:** `Q4_K_M`
- **Model File:** `gemma-4-E2B-it-Q4_K_M.gguf` (~2.96 GB)
- **SHA-256 Checksum:** `740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`
- **Architecture:** `Gemma4ForConditionalGeneration`
- **License:** `Apache-2.0` (Commercially permissive)
- **Execution Target:** CPU-first (`llama.cpp` / `llama-cpp-python` with OpenMP)
- **Container Port:** `8331` (internal private network only)

The RIAS LLM service is an internal-only microservice. It is never exposed directly to the public internet or through host port mappings. Only the API Gateway serves as the public ingress, while the Cloud Inference Orchestrator routes validated ASR transcripts to Gemma over `cfic-internal-network`.

---

## 2. Microservice Architecture & Data Flow

```text
10-second RIAS audio chunk
        │
        ▼
Language Identification (Open-Vakgyata)
        │
        ▼
Exact Language-Specific ASR (Whisper Base.en / IndicConformer)
        │
        ▼
Transcript Validation & Length Check
        │
        ▼
Gemma 4 E2B IT Microservice (POST /v1/assess via cfic-internal-network:8331)
        │
        ├── System Prompt Enclosure (<transcript> XML tags)
        ├── Single Model Instance (asyncio.Semaphore guarded)
        ├── Deterministic Sampling (temperature=0.0)
        └── Output Parser (Normalizes LOW, MEDIUM, or HIGH)
        │
        ▼
Normalized RIAS Result
        │
        ▼
TCED Decision Fusion (API Gateway)
```

---

## 3. Environment Variables Reference

| Variable | Default Value | Description |
|---|---|---|
| `GEMMA_MODEL_PATH` | `/app/weights/gemma-4-E2B-it-Q4_K_M.gguf` | Absolute path to mounted GGUF model file |
| `RIAS_MODEL_ID` | `google/gemma-4-E2B-it` | Model identifier string |
| `RIAS_MODEL_REVISION` | `4adf15748de9e1a0097b6274cee9c15836c41c37` | Model revision commit hash |
| `RIAS_MODEL_SHA256` | `740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8` | Expected SHA-256 checksum |
| `RIAS_REQUIRE_CHECKSUM`| `false` | When true, verifies SHA-256 on startup |
| `RIAS_USE_MOCK` | `false` | When true, enables mock mode for testing without GGUF weights |
| `RIAS_N_THREADS` | `4` | CPU inference thread allocation |
| `RIAS_N_CTX` | `2048` | Context window size |
| `RIAS_MAX_TOKENS` | `16` | Maximum generated tokens (single label output) |
| `RIAS_TIMEOUT_SECONDS` | `15.0` | Maximum inference timeout before failing to INCONCLUSIVE |
| `RIAS_QUEUE_TIMEOUT_SECONDS` | `5.0` | Maximum wait time in concurrency queue |
| `RIAS_MAX_CONCURRENT_REQUESTS` | `2` | Maximum concurrent inference requests |
| `RIAS_MAX_QUEUE_SIZE` | `10` | Maximum waiting requests before rejecting with QUEUE_OVERFLOW |
| `RIAS_MAX_TRANSCRIPT_LENGTH` | `5000` | Maximum characters permitted per transcript |

---

## 4. Internal API Contract

### Endpoints

- `GET /health` — Returns process status, model loaded state, device, and error logs.
- `GET /ready` — Readiness probe. Returns HTTP 200 `{"status": "ready"}` or HTTP 503 if model is unmounted/loading failed.
- `GET /metadata` — Architecture details, quantization, context size, and allowed output labels.
- `POST /v1/assess` — Semantic risk classification.

### Request Body
```json
{
  "job_id": "sess-prod-001",
  "chunk_index": 0,
  "language": "en",
  "transcript": "URGENT: This is State Bank fraud division. Your account has been suspended due to unauthorized activity. Read back your OTP to verify ownership."
}
```

### Successful Response (`status="SUCCESS"`)
```json
{
  "job_id": "sess-prod-001",
  "chunk_index": 0,
  "status": "SUCCESS",
  "risk_level": "HIGH",
  "failure_reason": null,
  "model_id": "google/gemma-4-E2B-it",
  "model_revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
  "latency_ms": 735.6
}
```

### Failure / Inconclusive Response (`status="INCONCLUSIVE"`)
```json
{
  "job_id": "sess-prod-001",
  "chunk_index": 0,
  "status": "INCONCLUSIVE",
  "risk_level": "INCONCLUSIVE",
  "failure_reason": "INFERENCE_TIMEOUT",
  "model_id": "google/gemma-4-E2B-it",
  "model_revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
  "latency_ms": 15000.0
}
```

---

## 5. Step-by-Step Deployment Instructions

### 1. Download Model Weights
```powershell
python cloud-inference/llm-evaluation/scripts/download_model.py unsloth/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_K_M.gguf
```

### 2. Verify Checksum
```powershell
# PowerShell
Get-FileHash cloud-inference/llm-evaluation/models/gemma-4-E2B-it-Q4_K_M.gguf -Algorithm SHA256
# Expected: 740185B21D22CEB83A11C3AA62AD5842EF32C70F6096D756BBEE85A1E4EC34B8
```

### 3. Build Docker Service Image
```bash
docker compose build model-rias-gemma
```

### 4. Start Compose Stack
```bash
docker compose up -d model-rias-gemma cloud-inference-orchestrator api-gateway
```

### 5. Check Service Readiness
```bash
docker exec -it cfic-inference-orchestrator curl -s http://model-rias-gemma:8331/ready
# Output: {"status":"ready","loaded":true,"model_id":"google/gemma-4-E2B-it","active_requests":0,"max_concurrency":2}
```

### 6. Run Sample Inference
```bash
docker exec -it cfic-inference-orchestrator curl -s -X POST http://model-rias-gemma:8331/v1/assess \
  -H "Content-Type: application/json" \
  -d '{"job_id":"test-1","chunk_index":0,"language":"en","transcript":"Digital arrest warrant issued. Transfer funds immediately."}'
```

### 7. Stop Services
```bash
docker compose down
```

---

## 6. Real CPU Performance Benchmark

Measurements taken on host hardware (**Intel Core i5-1334U**, 10 cores / 12 threads, 16 GB RAM) with actual GGUF `Q4_K_M` weights:

| Metric / Parameter | Empirical Result | Notes / Production Context |
|---|---:|---|
| **Cold Model Load Time** | **2.188 s** | Initialized once during container lifespan |
| **Idle Process RAM** | **2,343.2 MB** (~2.29 GB) | Baseline memory after model loading |
| **Peak Process RAM** | **2,808.2 MB** (~2.74 GB) | Well within the 4 GB container limit |
| **Median (P50) Latency** | **431.5 ms** | Sub-half-second typical decision time |
| **Average Latency** | **1,025.8 ms** (~1.03 s) | Across short, medium, and long transcripts |
| **P95 Latency** | **3,691.4 ms** (~3.69 s) | Worst-case cold prompt evaluation (< 5s limit) |
| **Warm Repeat Speed** | **~125 ms** (7.8–8.0 tok/s) | Cached prefix generation speed |
| **Mean CPU Utilization** | **70.2 %** | Measured during full batch CPU execution |
| **Concurrent Concurrency** | **2 active / 10 queue** | Prevents CPU starvation of ASR / Spoof |

---

## 7. Safety Invariants & Fallback Semantics

1. **Inconclusive Never Maps to LOW:** Any model crash, timeout, queue overflow, empty transcript, or unparsable token response yields `risk_level: "INCONCLUSIVE"`.
2. **TCED Decision Policy:** When RIAS produces `HIGH`, the API Gateway Decision Engine mandates an immediate `SUSPICIOUS` verdict (`HIGH_RIAS_OVERRIDE`), regardless of acoustic spoof detector confidence.
3. **Privacy Invariant:** Raw transcript strings are never printed in operational container logs.

> [!IMPORTANT]
> **Validation Caveat:** The performance metrics reported herein reflect empirical benchmarks on the reported 81-sample held-out evaluation dataset under controlled single-process CPU benchmark conditions (`Intel Core i5-1334U`, `n_threads=4`). Ground-truth label provenance, domain transfer resilience, and independent human annotation verification must be verified before treating these figures as definitive evidence of universal generalization.
