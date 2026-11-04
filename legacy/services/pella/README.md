# PellaV2 Deepfake Detector Service

**Assigned Owner:** Varshu (Cloud Engineer) in collaboration with Varun (MLOps Architect).  
**Target Port:** `8101`  
**Endpoints:** `GET /health`, `POST /predict`

---

## Service Overview

This microservice wraps the PellaV2 audio deepfake detector model behind an HTTP API.  
It is invoked concurrently alongside the transcription service by the CFIC API during call analysis.

### Service Contract

#### `GET /health`
Returns service availability, model version, whether mock mode is active, and artifact presence.

#### `POST /predict`
- **Request:** `multipart/form-data` with `file: UploadFile` (WAV audio recommended, 16 kHz mono).
- **Response Schema:**
```json
{
  "status": "success",
  "prediction": "GENUINE",
  "score": 0.15,
  "model_version": "pellav2-scaffold-v0.1",
  "latency_ms": 28.5
}
```

---

## Tasks for Varun & Varshu

1. **Verify Artifact (Varun):**
   - Confirm exact model architecture and revision.
   - Record class output mapping (e.g. 0 = Genuine, 1 = Fake or vice versa).
   - Evaluate quantization (FP16 / INT8 / ONNX).
   - Hand off artifact and `models/artifact_manifest.json` to Varshu.

2. **Implement Inference (Varshu):**
   - Replace mock logic in `service.py` with real model loading and forward pass.
   - Set `PELLA_USE_MOCK=false`.
   - Validate input audio sample rate conversion using `soundfile` or `ffmpeg`.
   - Measure startup time, memory consumption, and inference latency.

3. **Build & Test Docker (Varshu):**
```bash
docker build -f services/pella/Dockerfile -t cfic-pella-service:latest .
docker run -p 8101:8101 -e PELLA_USE_MOCK=true cfic-pella-service:latest
curl http://localhost:8101/health
```
