# RIAS Semantic Risk LLM Service Contract

## Overview
- **Model Checkpoint:** `google/gemma-4-E2B-it`
- **GGUF Source:** `unsloth/gemma-4-E2B-it-GGUF` (Commit SHA: `0314792d7f1f7e229411f620751375812bb9faf2`)
- **Quantization:** `Q4_K_M` (File: `gemma-4-E2B-it-Q4_K_M.gguf`, SHA-256: `740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`)
- **Architecture:** `Gemma4ForConditionalGeneration`
- **Runtime:** `llama.cpp` via `llama-cpp-python` (CPU-first execution)
- **License:** `Apache-2.0`
- **Internal Container Port:** `8331`
- **Network Scope:** `cfic-internal-network` only (private, zero public host exposure)

---

## Endpoints

### 1. `GET /health`
Returns runtime process health and model load state.

#### Response
```json
{
  "status": "ok",
  "service": "model-rias-gemma",
  "loaded": true,
  "model_id": "google/gemma-4-E2B-it",
  "model_revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
  "device": "cpu",
  "is_mock": false,
  "error": null
}
```

### 2. `GET /ready`
Readiness probe verifying that the model weights are loaded and active concurrency slots are available.
- Returns HTTP 200 `{"status": "ready", "loaded": true, ...}`
- Returns HTTP 503 `{"detail": "Gemma model is not ready..."}` if weights are unmounted or failed to load.

### 3. `GET /metadata`
Returns model specifications, quantization, context window, and thread allocations.

### 4. `POST /v1/assess`
Evaluates conversational audio transcripts for social engineering, financial coercion, impersonation, and fraud risk.

#### Request Schema
```json
{
  "job_id": "sess-12345",
  "chunk_index": 0,
  "language": "en",
  "transcript": "Hello, this is officer Sharma from Delhi Crime Branch. Your bank account is linked to money laundering and you are placed under digital arrest. Transfer funds immediately."
}
```

#### Successful Response Schema (`status="SUCCESS"`)
```json
{
  "job_id": "sess-12345",
  "chunk_index": 0,
  "status": "SUCCESS",
  "risk_level": "HIGH",
  "failure_reason": null,
  "model_id": "google/gemma-4-E2B-it",
  "model_revision": "4adf15748de9e1a0097b6274cee9c15836c41c37",
  "latency_ms": 3120.5
}
```

#### Failure / Inconclusive Response Schema (`status="INCONCLUSIVE"`)
```json
{
  "job_id": "sess-12345",
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

## Safety Invariants

1. **Strict Tri-State Output:** The only valid successful risk classifications are `LOW`, `MEDIUM`, or `HIGH`.
2. **Fail-Safe Inconclusive Policy:**
   - Any runtime failure, connection error, inference timeout, unclosed queue, or invalid token emission strictly maps to `risk_level: "INCONCLUSIVE"`.
   - **Never default to `LOW` on failure.**
3. **Empty Transcript Handling:** Empty or whitespace-only transcripts produce `status: "SKIPPED"` and `risk_level: "INCONCLUSIVE"`.
4. **Input Size Limits:** Transcripts exceeding `RIAS_MAX_TRANSCRIPT_LENGTH` (5,000 characters) are rejected with `TRANSCRIPT_OVERSIZED`.
5. **Prompt Injection Resistance:** Transcripts are treated strictly as untrusted user inputs wrapped in XML delimiters (`<transcript>...</transcript>`).
6. **Privacy Protection:** Raw conversational transcripts are never logged to console or stored in log files.
