# Language Identification (LID) Service Contract

## Overview
- **Model Checkpoint:** `onecxi/open-vakgyata`
- **Revision:** `f2754058e485dfc65cc62589b8d0e21c1d328399`
- **Architecture:** `Wav2Vec2ForSequenceClassification`
- **License:** `CC-BY-NC-4.0` (Non-Commercial Research & Evaluation)
- **Container Port:** `8301`
- **Network Scope:** `cfic-internal-network` only (isolated)

## Supported Languages (10 Indian & English Dialects)
| Canonical Code | Model Label (`id2label`) | Language Name |
|---|---|---|
| `en` | `en-IN` (0) | English (India) |
| `hi` | `hi-IN` (1) | Hindi |
| `or` | `or-IN` (2) | Odia |
| `bn` | `bn-IN` (3) | Bengali |
| `ta` | `ta-IN` (4) | Tamil |
| `te` | `te-IN` (5) | Telugu |
| `kn` | `kn-IN` (6) | Kannada |
| `ml` | `ml-IN` (7) | Malayalam |
| `mr` | `mr-IN` (8) | Marathi |
| `gu` | `gu-IN` (9) | Gujarati |

## Endpoints

### 1. `GET /health`
Returns service readiness and loaded checkpoint details.

### 2. `GET /ready`
Returns HTTP 200 `{"status": "ready"}` if weights are loaded; HTTP 503 if unavailable.

### 3. `GET /metadata`
Returns model architecture, commit revision, license, and class mappings.

### 4. `POST /identify`
Performs language classification on input audio.

#### Request
- `file`: `UploadFile` (multipart/form-data) — 16 kHz mono WAV recommended.
- `top_k`: `int` (query param, default: 5).

#### Response Schema
```json
{
  "status": "success",
  "language": "en",
  "raw_label": "en-IN",
  "confidence": 0.9998,
  "probabilities": {
    "en": 0.9998,
    "hi": 0.0001,
    "te": 0.0000,
    "ta": 0.0000,
    "kn": 0.0000,
    "ml": 0.0000,
    "mr": 0.0000,
    "gu": 0.0000,
    "bn": 0.0000,
    "or": 0.0000
  },
  "top_k": [
    {"language": "en", "raw_label": "en-IN", "confidence": 0.9998},
    {"language": "hi", "raw_label": "hi-IN", "confidence": 0.0001}
  ],
  "model_id": "onecxi/open-vakgyata",
  "model_revision": "f2754058e485dfc65cc62589b8d0e21c1d328399",
  "latency_ms": 550.2,
  "audio_duration_seconds": 8.84,
  "device": "cpu"
}
```

#### Special Conditions
- **Silence:** When audio RMS energy < `LID_SILENCE_RMS_THRESHOLD` (0.002), returns `status="silence"`, `language="unknown"`, `confidence=0.0`.
- **Malformed Audio:** Returns HTTP 400 with descriptive error detail.
