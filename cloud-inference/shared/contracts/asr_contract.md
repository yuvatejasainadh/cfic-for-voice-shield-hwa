# Automatic Speech Recognition (ASR) Service Contracts

## English ASR Service (`model-asr-en`)
- **Model Checkpoint:** `openai/whisper-base.en`
- **Revision:** `911407f4214e0e1d82085af863093ec0b66f9cd6`
- **Architecture:** `WhisperForConditionalGeneration` (English-only base model)
- **License:** `Apache-2.0`
- **Container Port:** `8311`
- **Network Scope:** `cfic-internal-network` only

### Endpoints

#### 1. `GET /health`
Returns service readiness and loaded checkpoint details.

#### 2. `GET /ready`
Returns HTTP 200 `{"status": "ready"}` if weights are loaded; HTTP 503 if unavailable.

#### 3. `GET /metadata`
Returns model architecture, commit revision, and license.

#### 4. `POST /transcribe`
Transcribes English speech to text.

##### Request
- `file`: `UploadFile` (multipart/form-data) — 16 kHz mono WAV.

##### Response Schema
```json
{
  "status": "success",
  "text": "The studies indicate that there is some utility in attempting to designate certain buildings as involving a higher risk than others.",
  "language": "en",
  "model_id": "openai/whisper-base.en",
  "model_revision": "911407f4214e0e1d82085af863093ec0b66f9cd6",
  "latency_ms": 1047.2,
  "audio_duration_seconds": 8.84,
  "device": "cpu"
}
```

##### Special Conditions
- **Silence:** When audio RMS energy < `ASR_EN_SILENCE_RMS_THRESHOLD` (0.002), returns `status="silence"`, `text=""`.
- **Malformed Audio:** Returns HTTP 400 with descriptive error detail.

---

## Indic Language ASR Services
- **Telugu (`model-asr-te`, Port 8312):** AI4Bharat IndicConformer Telugu (`indicconformer_stt_te_hybrid_rnnt_large.nemo`).
- **Hindi (`model-asr-hi`, Port 8313):** AI4Bharat IndicConformer Hindi (`indicconformer_stt_hi_hybrid_rnnt_large.nemo`).
