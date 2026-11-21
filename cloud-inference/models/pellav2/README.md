# PellaV2 Audio Deepfake Detection Microservice

Containerized acoustic deepfake detection service for CloudForge Intelligence Core (CFIC) × VoiceShield.

## Model Details
- **Source:** Hugging Face `Sadanie/pellav2-audio-deepfake-detector`
- **Revision:** `94b899915ee551fa90d8844961532e4aaab58322`
- **Weights File:** `weights/pellav2_detector.pt` (SHA-256: `d21aedfb52e2f87706c2753715a29665482b0f4a034800c31579df5d458559ac`)
- **Offline Config:** `config/w2v_bert_config.json`
- **Backbone:** `facebook/w2v-bert-2.0` (300M parameter Conformer)
- **Precision:** FP32 (Full precision)

## Endpoints
- `GET /health` - Liveness probe & SHA-256 verification
- `GET /ready` - Readiness probe (HTTP 200 when weights are ready, 503 otherwise)
- `GET /metadata` - Model parameters and configuration
- `POST /predict` - Upload 16kHz mono WAV audio multipart form-data

## Running Locally
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8321
```
