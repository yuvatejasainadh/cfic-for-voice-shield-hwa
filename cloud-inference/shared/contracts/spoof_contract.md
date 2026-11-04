# Acoustic Spoof Detection Services Contract

**Base URLs:**
- VMC-South (Fine-tuned AASIST-L for South Indian Speech): `http://model-vmc-south:8321`
- AASIST-L Fallback / English: `http://model-aasist-fallback:8322`

---

## 1. `GET /health`
```json
{
  "status": "ok",
  "service": "model-vmc-south",
  "architecture": "AASIST-L-FineTuned",
  "target_domain": "South-Indian-Multilingual",
  "model_version": "vmc-south-v1.0",
  "artifact_found": true
}
```

---

## 2. `POST /predict`
- **Request:** `multipart/form-data` with `file: UploadFile` (16 kHz mono WAV)
- **Response Schema:**
```json
{
  "status": "success",
  "prediction": "GENUINE",
  "score": 0.042,
  "threshold": 0.50,
  "model_version": "vmc-south-v1.0",
  "latency_ms": 38.5
}
```
*Polarity: Score > threshold indicates synthetic/fake voice.*
