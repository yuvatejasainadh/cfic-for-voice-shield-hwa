# CFIC × VoiceShield HackVersion — Team Task Assignments

**Team SAIL for HackWithAditya**

This document specifies the decoupled task boundaries, deliverables, and acceptance checklists for each of the four team members.

---

## 1. Sainadh — Project Lead & Systems Architect

### Responsibilities
- Architect the end-to-end CFIC pipeline and VoiceShield integration.
- Implement API gateway (`/health`, `/v1/analyze`), API key authentication, and rate limiting.
- Implement parallel orchestration (`PellaV2` + `RIAS` transcription concurrently via `asyncio.gather`).
- Integrate hosted AskDrip LLM API for conversational risk classification.
- Implement TCED hybrid decision logic and enforce the mandatory `RIAS HIGH -> SUSPICIOUS` safety rule.
- Lead end-to-end testing, PR reviews, and merging into `main`.

### Checklist & Deliverables
- [x] FastAPI application scaffold with versioned routes (`/v1`).
- [x] Request ID tracking middleware attached to requests and response headers.
- [x] In-memory rate limiting middleware with configurable requests/window.
- [x] Request size validation (25 MiB max upload limit) and empty audio guard.
- [x] Pure TCED decision engine with mandatory HIGH override and fallback rules.
- [x] Async HTTP clients for PellaV2, Transcription, and AskDrip LLM with bounded timeouts.
- [ ] Connect production VoiceShield mobile/desktop client to `/v1/analyze`.
- [ ] Review and merge contributor pull requests from Varun, Varshu, and Pallavi.

---

## 2. Varun — MLOps Architect (PellaV2 Model Preparation)

### Responsibilities
- Inspect and verify the exact available PellaV2 model artifact (PyTorch checkpoint, ONNX, or Hugging Face revision).
- Determine native inference interface: input sampling rate (16 kHz mono), tensor shape, and normalization requirements.
- Identify exact class-score semantics:
  - Label mapping: which class is index 0 vs index 1?
  - Score polarity: does higher score indicate synthetic/fake, or genuine human voice?
- Quantize the model using technically sound techniques (e.g. INT8 dynamic quantization via ONNX Runtime or PyTorch Quantization).
- Benchmark original vs quantized model:
  - Model binary size (MB)
  - Runtime RAM footprint (MB)
  - Inference latency per 5-second audio chunk (ms)
  - Detection performance (EER / AUC / Accuracy) on validation audio.
- Complete the Artifact Manifest and hand off the validated package to Varshu.

### Working Workspace
- `docs/templates/artifact_manifest_template.json`
- `docs/templates/quantization_report_template.md`
- Recommended Feature Branch: `feature/varun-pellav2-model-preparation`

### Checklist & Deliverables
- [ ] Inspect raw PellaV2 checkpoint and record checksum (SHA-256).
- [ ] Verify if XLS-R backbone is embedded in the artifact or requires external download.
- [ ] Document verified output label mapping and score polarity.
- [ ] Execute quantization script and export quantized artifact (e.g., `pellav2_int8.onnx`).
- [ ] Fill out `docs/models/pellav2_artifact_manifest.json`.
- [ ] Fill out `docs/benchmarks/pellav2_quantization_report.md`.
- [ ] Hand off artifact, inference snippet, and dependencies to Varshu.
- [ ] Submit PR targeting `main`.

---

## 3. Varshu — Cloud Engineer (PellaV2 Docker Microservice)

### Responsibilities
- Containerize the validated PellaV2 inference service using Docker.
- Implement standalone microservice in `services/pella/service.py` running on port `8101`.
- Expose `GET /health` and `POST /predict` adhering strictly to `docs/API_CONTRACTS.md`.
- Implement robust audio input validation (checking file presence, MIME types, and sample rate conversion).
- Load the validated artifact handed off by Varun (keeping mock fallback active if artifact is missing).
- Measure cold-start latency, memory usage, and batch inference performance in Docker.
- Provide build and run documentation.

### Working Workspace
- `services/pella/`
- Recommended Feature Branch: `feature/varshu-pellav2-docker-service`

### Checklist & Deliverables
- [ ] Implement `services/pella/service.py` with real model inference (replacing mock).
- [ ] Ensure `POST /predict` accepts `UploadFile` and returns normalized JSON.
- [ ] Configure `services/pella/Dockerfile` with required audio libraries (`libsndfile1`, `ffmpeg`).
- [ ] Pin dependencies in `services/pella/requirements.txt` (Torch/ONNX Runtime).
- [ ] Add service contract unit test in `tests/contracts/test_pella_contract.py`.
- [ ] Document container startup time and memory footprint in `services/pella/README.md`.
- [ ] Submit PR targeting `main`.

---

## 4. Pallavi — Cloud Engineer (Multilingual Transcription Microservice)

### Responsibilities
- Empirically benchmark Whisper Multilingual (e.g. `whisper-small` or `faster-whisper`) against AI4Bharat IndicConformer.
- Evaluate candidate target languages: English, Telugu, Tamil, Kannada, Malayalam.
- Measure and compare:
  - Transcription accuracy (Word Error Rate / Character Error Rate)
  - Inference latency per audio second
  - Model size and RAM footprint
  - Deployment feasibility (dependencies, license, container stability).
- Package the recommended model into a standalone Docker service in `services/transcription/`.
- Expose `GET /health` and `POST /transcribe` adhering strictly to `docs/API_CONTRACTS.md`.
- Handle audio resampling and noise resilience.

### Working Workspace
- `services/transcription/`
- `docs/templates/transcription_evaluation_template.md`
- Recommended Feature Branch: `feature/pallavi-transcription-service`

### Checklist & Deliverables
- [ ] Run transcription benchmark across candidate languages using test audio samples.
- [ ] Complete `docs/benchmarks/transcription_evaluation_report.md` with empirical WER/latency numbers.
- [ ] Make explicit, evidence-backed model recommendation (Whisper vs IndicConformer).
- [ ] Implement `services/transcription/service.py` with the selected model runtime.
- [ ] Configure `services/transcription/Dockerfile` and pin `requirements.txt`.
- [ ] Verify `POST /transcribe` returns normalized text, detected language, and latency.
- [ ] Add service contract unit test in `tests/contracts/test_transcription_contract.py`.
- [ ] Submit PR targeting `main`.
