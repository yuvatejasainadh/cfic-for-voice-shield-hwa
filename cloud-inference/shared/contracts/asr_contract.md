# IndicConformer ASR Services Contract

**Base URLs:**
- English: `http://model-asr-en:8311`
- Telugu: `http://model-asr-te:8312`
- Hindi: `http://model-asr-hi:8313`

---

## 1. `GET /health`
```json
{
  "status": "ok",
  "service": "model-asr-en",
  "target_language": "en",
  "model_version": "indicconformer-en-large-v1",
  "artifact_found": true
}
```

---

## 2. `POST /transcribe`
- **Request:** `multipart/form-data` with `file: UploadFile`
- **Response Schema:**
```json
{
  "status": "success",
  "text": "Please confirm your security details immediately.",
  "language": "en",
  "model_version": "indicconformer-en-large-v1",
  "decoder": "ctc",
  "latency_ms": 115.0
}
```
