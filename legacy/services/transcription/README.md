# Multilingual Transcription Service

**Assigned Owner:** Pallavi (Cloud Engineer).  
**Target Port:** `8102`  
**Endpoints:** `GET /health`, `POST /transcribe`

---

## Service Overview

This microservice handles automatic speech recognition (ASR) for incoming call audio.  
The resulting transcript is forwarded to the hosted AskDrip LLM API (RIAS Stage 2) for conversational risk assessment.

### Service Contract

#### `GET /health`
Returns service status, model version, whether mock mode is active, and artifact presence.

#### `POST /transcribe`
- **Request:** `multipart/form-data` with `file: UploadFile` (WAV/MP3 audio).
- **Response Schema:**
```json
{
  "status": "success",
  "text": "Hello, this is an urgent security call from your bank.",
  "language": "en",
  "model_version": "whisper-indic-mock-v0.1",
  "latency_ms": 42.1
}
```

---

## Tasks for Pallavi

1. **Benchmark Whisper Multilingual vs AI4Bharat IndicConformer:**
   - Target languages: English, Telugu, Tamil, Kannada, Malayalam.
   - Test clean, noisy, and code-switched audio samples.
   - Record WER/CER, memory footprint, model size, and inference latency per language.
   - Complete `docs/benchmarks/transcription_evaluation_report.md`.

2. **Package Selected Model:**
   - Integrate model into `service.py` (replacing the mock when `TRANSCRIPTION_USE_MOCK=false`).
   - Pin requirements in `services/transcription/requirements.txt`.
   - Ensure audio preprocessing (resampling to 16 kHz) is robust.

3. **Build & Run Docker Container:**
```bash
docker build -f services/transcription/Dockerfile -t cfic-transcription-service:latest .
docker run -p 8102:8102 -e TRANSCRIPTION_USE_MOCK=true cfic-transcription-service:latest
curl http://localhost:8102/health
```
