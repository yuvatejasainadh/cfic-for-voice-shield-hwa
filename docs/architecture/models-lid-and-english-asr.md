# Language Identification & English ASR Model Documentation

## 1. Overview

This document specifies the containerized model integrations for the CFIC Cloud Inference subsystem:
1. **Language Identification (LID):** `onecxi/open-vakgyata`
2. **English Automatic Speech Recognition (ASR):** `openai/whisper-base.en`

Both services run as isolated microservices within the internal Docker network (`cfic-internal-network`), coordinated by the Cloud Inference Orchestrator.

---

## 2. Open-Vakgyata Language Identification (LID)

### Model Identity & Licensing
- **Model Identifier:** `onecxi/open-vakgyata`
- **Hugging Face URL:** [https://huggingface.co/onecxi/open-vakgyata](https://huggingface.co/onecxi/open-vakgyata)
- **Immutable Commit SHA:** `f2754058e485dfc65cc62589b8d0e21c1d328399`
- **Architecture:** `Wav2Vec2ForSequenceClassification`
- **Model Size:** ~95M parameters (~380 MB `model.safetensors`)
- **License:** **`CC-BY-NC-4.0`** (Creative Commons Attribution Non-Commercial 4.0 International)

> [!CAUTION]
> **Commercial Licensing Constraint:** `onecxi/open-vakgyata` is licensed under `CC-BY-NC-4.0`, which strictly prohibits commercial exploitation without express authorization from OneCXI. For hackathon and research evaluation, this license is compliant. If CFIC is deployed commercially, an alternative model (or commercially cleared weights) must be evaluated and approved.

### Class Labels & Canonical Mapping
Vakgyata outputs probabilities across 10 classes (`id2label`), normalized into the project's canonical ISO 639-1 language codes:

| Class Index | Raw Model Label | Canonical Code | Language Name |
|---|---|---|---|
| 0 | `en-IN` | `en` | English (India) |
| 1 | `hi-IN` | `hi` | Hindi |
| 2 | `or-IN` | `or` | Odia |
| 3 | `bn-IN` | `bn` | Bengali |
| 4 | `ta-IN` | `ta` | Tamil |
| 5 | `te-IN` | `te` | Telugu |
| 6 | `kn-IN` | `kn` | Kannada |
| 7 | `ml-IN` | `ml` | Malayalam |
| 8 | `mr-IN` | `mr` | Marathi |
| 9 | `gu-IN` | `gu` | Gujarati |

### Service Endpoints & Port
- **Port:** `8301`
- **Endpoints:**
  - `GET /health` — Service readiness, device, model version.
  - `GET /ready` — Health probe returning HTTP 200/503.
  - `GET /metadata` — Model metadata, license, labels.
  - `POST /identify` — Multi-part audio payload (`UploadFile`), returns top canonical language, confidence score, raw label, and ranked `top_k` probabilities.

---

## 3. Dedicated English ASR: Whisper Base English

### Model Identity & Licensing
- **Model Identifier:** `openai/whisper-base.en`
- **Hugging Face URL:** [https://huggingface.co/openai/whisper-base.en](https://huggingface.co/openai/whisper-base.en)
- **Immutable Commit SHA:** `911407f4214e0e1d82085af863093ec0b66f9cd6`
- **Architecture:** `WhisperForConditionalGeneration` (English-only base encoder-decoder)
- **Model Size:** ~74M parameters (~290 MB `model.safetensors`)
- **License:** **`Apache-2.0`** (Commercially permissive)

### Service Endpoints & Port
- **Port:** `8311`
- **Endpoints:**
  - `GET /health` — Service readiness, device, model version.
  - `GET /ready` — Health probe returning HTTP 200/503.
  - `GET /metadata` — Model metadata, architecture, license.
  - `POST /transcribe` — Multi-part audio payload (`UploadFile`), returns decoded English text, latency, duration.

---

## 4. Empirical Evaluation & Benchmark Results

All evaluations were executed on the local host machine (Intel Core i5-1334U CPU, 16 GB RAM) against real speech samples from the Multilingual Speech Dataset (`C:\Users\yuvat\Music\Multilingual Dataset`):

### Open-Vakgyata LID Evaluation Summary
- **English Speech:** 100% precision (3/3 samples correctly classified as `en-IN`, confidence 1.0000, avg latency: **555.4ms** on CPU).
- **Tamil Speech:** 100% precision (3/3 samples correctly classified as `ta-IN`, confidence 1.0000, avg latency: **324.5ms**).
- **Marathi Speech:** 100% precision (3/3 samples correctly classified as `mr-IN`, confidence 0.9998, avg latency: **743.2ms**).
- **Gujarati Speech:** 100% precision (3/3 samples correctly classified as `gu-IN`, confidence 0.9658, avg latency: **667.6ms**).
- **Kannada & Bengali:** 67% accuracy (avg latency: **312ms** – **648ms**).
- **Silence & Short Audio Edge Cases:** Pure silence (2.0s zeros) produces `status="silence"`, `confidence=0.0`, safely preventing false downstream invocation.

### OpenAI Whisper-Base.en Evaluation Summary
- **Dataset:** 10 representative test utterances (75.34s total audio duration) from `test.csv`.
- **Word Error Rate (WER):** **10.93%**
- **Character Error Rate (CER):** **3.63%**
- **Real-Time Factor (RTF):** **0.139** (Processing 75.34 seconds of speech in 10.47 seconds on CPU — **7.2x faster than real-time**).
- **Average Latency:** **1047.2ms** per utterance (~7.5s speech).

---

## 5. Orchestrator Routing & Safety Rules

1. **Sequential RIAS Pipeline:**
   `Audio Chunk (10s) ➔ Open-Vakgyata LID ➔ Language Routing ➔ Exact Language ASR`
2. **Confidence Thresholding:**
   - Default threshold: `INFERENCE_LID_CONFIDENCE_THRESHOLD=0.60`.
   - Up to 2 retries on downstream transient errors (`INFERENCE_LID_MAX_RETRIES=2`).
3. **Inconclusive Circuit Breaking:**
   - If LID confidence is below 0.60 after retries or returns `"silence"` / `"error"`, the orchestrator marks the result as `INCONCLUSIVE`.
   - **ASR is explicitly SKIPPED** (`transcription.status = "skipped"`) — preventing hallucinated transcripts.
4. **Session Metadata Preservation:**
   - Every inference response preserves `session_id`, `chunk_index`, `timestamp_start_s`, and `timestamp_end_s`.
