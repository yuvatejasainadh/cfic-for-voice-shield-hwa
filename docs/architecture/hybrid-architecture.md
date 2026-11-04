# CFIC × VoiceShield — HYBRID Architecture Specification

**Repository:** `CFIC-VoiceShield-HackVersion`  
**Architecture Model:** Decoupled Hybrid Architecture (API Gateway + Cloud Inference)  
**Status:** Approved & Implemented

---

## 1. System Overview

**CloudForge Intelligence Core (CFIC)** is the backend intelligence system powering **VoiceShield**, designed for real-time defense against deepfake synthesized voices and conversational social engineering.

The system is strictly divided into two decoupled subsystems:
1. **API Gateway (`api-gateway/`)**: Public edge gateway managing client connections, authentication, sliding-window rate limiting, audio payload validation, 5-second chunking orchestration, application-level risk aggregation, decision management, and user-facing responses.
2. **Cloud Inference (`cloud-inference/`)**: Dedicated, model-serving subsystem managing Language Identification (LID), language-specific IndicConformer ASR checkpoints, and dual acoustic spoof detection models (VMC-South fine-tuned AASIST-L and AASIST-L fallback).

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
                │  - 5-Second Window Chunk Orchestration  │
                └────────────────────┬────────────────────┘
                                     │ (Private Network)
                                     ▼
                ┌─────────────────────────────────────────┐
                │     Cloud Inference Orchestrator (8200) │
                │  - Step 1: Language Identification      │
                │  - Step 2: Dynamic Checkpoint Routing   │
                │  - Step 3: Concurrent Model Execution   │
                └──────────────┬───────────────────┬──────┘
                               │                   │
                 ┌─────────────┴─────┐       ┌─────┴──────────────┐
                 ▼                   ▼       ▼                    ▼
          ┌─────────────┐     ┌─────────────┐ ┌────────────┐ ┌─────────────┐
          │  Model LID  │     │ Indic-      │ │ VMC-South  │ │  AASIST-L   │
          │   (8301)    │     │ Conformer   │ │ (AASIST-L  │ │  Fallback   │
          │   Identify  │     │ (EN/TE/HI)  │ │ South-Ind) │ │  (English)  │
          │   Language  │     │ (8311-8313) │ │   (8321)   │ │   (8322)    │
          └─────────────┘     └─────────────┘ └────────────┘ └─────────────┘
                                     │                   │
                                     └─────────┬─────────┘
                                               │ (Unified Inference Response)
                                               ▼
                ┌─────────────────────────────────────────┐
                │      API Gateway Decision Engine        │
                │  - Conversational Risk Synthesis (RIAS) │
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
- Temporary audio storage and 5-second chunking orchestration.
- Application-level job creation and status tracking.
- Semantic risk synthesis and conversational fraud analysis.
- Decision engine (fusing acoustic spoof verdict with conversational risk).
- Response normalization and client serialization.

### Cloud Inference Owns:
- Language Identification (LID).
- ASR model selection (English vs regional Indic checkpoints).
- IndicConformer CTC/RNNT transcription execution.
- Regional spoof detector routing:
  - **VMC-South**: Fine-tuned AASIST-L for South Indian speech (Telugu, Tamil, Kannada, Malayalam).
  - **AASIST-L Fallback**: Standard AASIST-L for English and non-regional audio.
- Model-specific audio resampling (16 kHz mono) and tensor preparation.
- Inference latency tracking per model.
- Model result normalization.

---

## 3. Decision Engine & Locked Safety Rules

The Decision Engine (`api-gateway/app/orchestration/decision_engine.py`) enforces strict safety guarantees:
1. **Locked Safety Override:** If conversational analysis returns `HIGH` risk, the final outcome **MUST BE `SUSPICIOUS`** (`HIGH_RIAS_OVERRIDE`), regardless of acoustic score.
2. **Fail-Safe Fallback:** If upstream model services fail, time out, or produce missing evidence, the system returns `INCONCLUSIVE`. It **never silently classifies an error as `GENUINE`**.
3. **Invalid Risk Guard:** Malformed or unrecognized risk levels immediately trigger `INCONCLUSIVE` (`INVALID_RIAS_RISK_LEVEL`).
