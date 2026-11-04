# Language Identification (LID) Service Contract

**Base URL:** `http://model-lid:8301`  
**Purpose:** Rapid language identification for incoming audio to route to appropriate ASR checkpoint and regional spoof detector.

---

## 1. `GET /health`
```json
{
  "status": "ok",
  "service": "model-lid",
  "model_version": "lid-ecapa-v1.0",
  "artifact_found": true
}
```

---

## 2. `POST /identify`
- **Request:** `multipart/form-data` with `file: UploadFile`
- **Response Schema:**
```json
{
  "status": "success",
  "language": "en",
  "confidence": 0.985,
  "supported_languages": ["en", "te", "hi", "ta", "kn", "ml"],
  "model_version": "lid-ecapa-v1.0",
  "latency_ms": 14.5
}
```
