# CFIC × VoiceShield — HYBRID Architecture Specification

**Repository:** `CFIC-VoiceShield-HackVersion`  
**Architecture Model:** Decoupled Hybrid Architecture (API Gateway + Cloud Inference)  
**Status:** Approved & Implemented

---

## 1. System Overview

**CloudForge Intelligence Core (CFIC)** is the backend intelligence system powering **VoiceShield**, designed for real-time defense against deepfake synthesized voices and conversational social engineering.

The system is strictly divided into two decoupled subsystems:
1. **API Gateway (`api-gateway/`)**: Public edge gateway managing client connections, authentication, sliding-window rate limiting, audio payload validation, chunking orchestration, application-level risk aggregation, decision management, and user-facing responses.
2. **Cloud Inference (`cloud-inference/`)**: Dedicated, model-serving subsystem managing Language Identification (LID), language-specific ASR checkpoints (English Whisper Base.en & IndicConformer), dual acoustic spoof detection models (VMC-South and AASIST-L fallback), and containerized Gemma 4 E2B IT RIAS LLM inference.

```
                      VoiceShield Client Audio Payload
                                     │
                                     ▼ (HTTP POST /v1/analyze, X-API-Key)
                ┌─────────────────────────────────────────┐
                │          CFIC API Gateway (8000)        │
                │  - X-API-Key Authentication             │
                │  - Sliding Window Rate Limiter (30/60s) │
                │  - Max Audio Validation (≤ 25 MiB)      │
                │  - X-Request-ID Distributed Tracing     │
                │  - 10-Second Window Chunk Orchestration │
                └────────────────────┬────────────────────┘
                                     │ (Private Network)
                                     ▼
                ┌─────────────────────────────────────────┐
                │     Cloud Inference Orchestrator (8200) │
                │  - Step 1: Language Identification      │
                │  - Step 2: Dynamic Checkpoint Routing   │
                │  - Step 3: Concurrent Acoustic Spoof    │
                │  - Step 4: Gemma RIAS Semantic Risk     │
                └──────────────┬───────────────────┬──────┘
                               │                   │
                 ┌─────────────┴─────┐       ┌─────┴──────────────┐
                 ▼                   ▼       ▼                    ▼
          ┌─────────────┐     ┌─────────────┐ ┌────────────┐ ┌─────────────┐
          │  Model LID  │     │ Indic-      │ │ VMC-South  │ │  AASIST-L   │
          │   (8301)    │     │ Conformer / │ │ (AASIST-L  │ │  Fallback   │
          │   Identify  │     │ Whisper EN  │ │ South-Ind) │ │  (English)  │
          │   Language  │     │ (8311-8313) │ │   (8321)   │ │   (8322)    │
          └─────────────┘     └──────┬──────┘ └────────────┘ └─────────────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │ Model RIAS  │
                              │ Gemma 4 E2B │
                              │   (8331)    │
                              └──────┬──────┘
                                     │ (Unified Multi-Modal Response)
                                     ▼
                ┌─────────────────────────────────────────┐
                │      API Gateway Decision Engine        │
                │  - Acoustic + Semantic Fusion (TCED)    │
                │  - Mandatory HIGH Risk Override         │
                │  - Fail-Safe Inconclusive Policy        │
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │            Client Response              │
                │   (GENUINE / FAKE / SUSPICIOUS /        │
                │            INCONCLUSIVE)                │
                └─────────────────────────────────────────┘
```

---

## 2. Architectural Boundaries & Responsibility Separation

### API Gateway Owns:
- Client-facing REST API endpoints (`POST /v1/analyze`, `GET /health`).
- Authentication and client authorization (`X-API-Key`).
- Sliding-window in-memory rate limiting.
- Audio payload validation (MIME types, size cap ≤ 25 MiB).
- Temporary audio storage and 10-second chunking orchestration.
- Application-level job creation and status tracking.
- Decision engine (fusing acoustic spoof verdict with conversational risk via TCED).
- Response normalization and client serialization.

### Cloud Inference Owns:
- Language Identification (LID) via `onecxi/open-vakgyata` (port 8301).
- ASR model selection (English `openai/whisper-base.en` on port 8311 vs regional IndicConformer checkpoints).
- Regional spoof detector routing:
  - **VMC-South**: Fine-tuned AASIST-L for South Indian speech (Telugu, Tamil, Kannada, Malayalam, port 8321).
  - **AASIST-L Fallback**: Standard AASIST-L for English and non-regional audio (port 8322).
- Semantic fraud risk assessment via containerized **Gemma 4 E2B IT Q4_K_M** (port 8331).
- Model-specific audio resampling (16 kHz mono) and tensor preparation.
- Inference latency tracking per model.
- Model result normalization.

---

## 3. Decision Engine & Locked Safety Rules

The Decision Engine (`api-gateway/app/orchestration/decision_engine.py`) enforces strict safety guarantees:
1. **Locked Safety Override:** If conversational analysis returns `HIGH` risk, the final outcome **MUST BE `SUSPICIOUS`** (`HIGH_RIAS_OVERRIDE`), regardless of acoustic score.
2. **Fail-Safe Fallback:** If upstream model services fail, time out, or produce missing evidence, the system returns `INCONCLUSIVE`. It **never silently classifies an error as `GENUINE`**.
3. **Invalid Risk Guard:** Malformed or unrecognized risk levels immediately trigger `INCONCLUSIVE` (`INVALID_RIAS_RISK_LEVEL`).
