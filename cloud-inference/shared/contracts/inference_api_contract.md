# Cloud Inference Orchestrator — API Contract

**Base URL:** `http://cloud-inference-orchestrator:8200`  
**Purpose:** Central entry point for all ML model executions. Decoupled from user auth, payments, database, or client sessions.

---

## 1. `GET /health`
Verifies readiness of orchestrator and availability of downstream model backends.

- **Response (200 OK):**
```json
{
  "status": "ok",
  "service": "cloud-inference-orchestrator",
  "version": "0.1.0",
  "backends": {
    "lid_service": "configured",
    "indicconformer_english": "configured",
    "indicconformer_telugu": "configured",
    "indicconformer_hindi": "configured",
    "vmc_south": "configured",
    "aasist_fallback": "configured"
  }
}
```

---

## 2. `POST /v1/inference`
Main batch inference endpoint invoked by API Gateway `CFICInferenceClient`.

- **Headers:** `X-Request-ID` (optional, forwarded)
- **Request:** `multipart/form-data` with `file: UploadFile` (16 kHz mono WAV recommended)
- **Response Schema:**
```json
{
  "status": "success",
  "lid": {
    "status": "success",
    "language": "en",
    "confidence": 0.98,
    "model_version": "lid-ecapa-v1.0",
    "latency_ms": 15.2
  },
  "transcription": {
    "status": "success",
    "text": "Hello, this is an urgent security verification.",
    "language": "en",
    "model_version": "indicconformer-en-large-v1",
    "latency_ms": 112.5
  },
  "spoof": {
    "status": "success",
    "prediction": "GENUINE",
    "score": 0.08,
    "model_version": "aasist-fallback-v1",
    "latency_ms": 42.0
  },
  "latency_ms": 135.0
}
```
