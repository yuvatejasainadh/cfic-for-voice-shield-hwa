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

## Validated Performance Benchmarks & Limitations
- **120-Sample Dataset Benchmark:** 71.67% accuracy, 80.95% precision, 56.67% recall, 0.8483 ROC-AUC, 13.33% FPR.
- **240-Sample Expanded Benchmark:** 66.67% accuracy, 73.81% precision, 51.67% recall, 0.7557 ROC-AUC, 18.33% FPR.
- **24-Sample Live Docker Spot Check:** 70.83% accuracy, 85.71% precision, 50.00% recall, 8.33% FPR, 50.00% FNR.
- **Known Limitation:** PellaV2 exhibits substantial false negatives (~45–50% FNR) when synthetic speech is blended with environmental or synthetic background noise due to acoustic energy masking. It must be operated in combination with conversational risk (RIAS) and the complete TCED fusion policy.

## Running Locally
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8321
```
