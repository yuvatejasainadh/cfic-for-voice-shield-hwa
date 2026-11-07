# CFIC × VoiceShield — HYBRID Architecture Specification

**Repository:** `CFIC-VoiceShield-HackVersion`  
**Architecture Model:** Decoupled Hybrid Architecture (API Gateway + Cloud Inference)  
**Status:** Approved, Implemented & Fully Validated

---

## 1. System Overview

**CloudForge Intelligence Core (CFIC)** is the backend intelligence system powering **VoiceShield**, designed for real-time defense against deepfake synthesized voices and conversational social engineering.

The system is strictly divided into two decoupled subsystems:
1. **API Gateway (`api-gateway/`)**: Public edge gateway managing client connections, authentication, sliding-window rate limiting, audio payload validation, 5-second chunking orchestration, application-level risk aggregation, decision management, and user-facing responses.
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
                │  - 5-Second Window Chunk Orchestration  │
                └────────────────────┬────────────────────┘
                                     │ (Docker Internal Network)
                                     ▼
                ┌─────────────────────────────────────────┐
                │     Cloud Inference Orchestrator (8200) │
                │  - Step 1: Open-Vakgyata LID (8301)     │
                │  - Step 2: Language-Specific ASR Routing│
                │    * 'te' -> IndicConformer Telugu(8312)│
                │    * 'hi' -> IndicConformer Hindi (8313)│
                │    * 'en' -> Whisper Base.en (8311)     │
                │    * Other -> Unsupported (skip ASR)    │
                │  - Step 3: Dual Acoustic Spoof Detector │
                │    * South Indian -> VMC-South (8321)   │
                │    * English/General -> AASIST-L (8322) │
                │  - Step 4: Gemma 4 E2B IT RIAS (8331)   │
                └──────────────┬───────────────────┬──────┘
                               │                   │
                 ┌─────────────┴─────┐       ┌─────┴──────────────┐
                 ▼                   ▼       ▼                    ▼
          ┌─────────────┐     ┌─────────────┐ ┌────────────┐ ┌─────────────┐
          │  Model LID  │     │ Indic-      │ │ VMC-South  │ │  AASIST-L   │
          │ (Vakgyata)  │     │ Conformer / │ │ (AASIST-L  │ │  Fallback   │
          │   (8301)    │     │ Whisper EN  │ │ South-Ind) │ │  (English)  │
          │  Identify   │     │ (8311-8313) │ │   (8321)   │ │   (8322)    │
          └─────────────┘     └──────┬──────┘ └────────────┘ └─────────────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │ Model RIAS  │
                              │ Gemma 4 E2B │
                              │   (8331)    │
                              └──────┬──────┘
                                     │ (Multi-Modal Unified Result)
                                     ▼
                ┌─────────────────────────────────────────┐
                │      API Gateway Decision Engine        │
                │  - Acoustic + Semantic Fusion (TCED)    │
                │  - Sequential Transcript Aggregation    │
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
- Client-facing REST API endpoints (`POST /v1/analyze`, `GET /health`, `GET /v1/health`).
- Authentication and client authorization (`X-API-Key`).
- Sliding-window in-memory rate limiting (30 req / 60s per client).
- Audio payload validation (MIME types, size cap ≤ 25 MiB).
- 5-second audio chunking orchestration via `soundfile` / `wave` (`slice_into_5s_chunks`).
- Sequential transcript assembly across audio chunks without transcript loss.
- Decision engine (fusing acoustic spoof verdict with conversational risk via TCED).
- Response normalization and client serialization.

### Cloud Inference Owns:
- Language Identification (LID) via `onecxi/open-vakgyata` (port 8301).
- Language-specific ASR routing without silent substitution:
  - Telugu (`te`) -> `ai4bharat/indicconformer-te` (port 8312).
  - Hindi (`hi`) -> `ai4bharat/indicconformer-hi` (port 8313).
  - English (`en`) -> `openai/whisper-base.en` (port 8311).
  - Unsupported languages -> Explicit `unsupported_language` status; ASR and RIAS skipped.
- Dual regional acoustic spoof detection:
  - **VMC-South**: Fine-tuned AASIST-L for South Indian speech (`te`, `ta`, `kn`, `ml`, port 8321).
  - **AASIST-L Fallback**: Standard AASIST-L for English and general audio (port 8322).
- Semantic fraud risk assessment via containerized **Gemma 4 E2B IT Q4_K_M** (port 8331).
- Internal network DNS communication (`cfic-internal-network`), with only `api-gateway` exposing public port 8000.

---

## 3. Decision Engine & Locked Safety Rules

The Decision Engine (`api-gateway/app/tced/decision_policy.py`) enforces strict safety guarantees:
1. **Locked Safety Override:** If conversational analysis returns `HIGH` risk, the final outcome **MUST BE `SUSPICIOUS`** (`HIGH_RIAS_OVERRIDE`), regardless of acoustic detector score.
2. **Fail-Safe Fallback:** If upstream model services fail, time out, or produce missing evidence, the system returns `INCONCLUSIVE`. It **never silently classifies an error as `GENUINE`**.
3. **Invalid Risk Guard:** Malformed or unrecognized risk levels immediately trigger `INCONCLUSIVE` (`INVALID_RIAS_RISK_LEVEL`).
4. **Unsupported Language Guard:** Unsupported spoken languages skip ASR and RIAS and yield `INCONCLUSIVE` (`PARTIAL_EVIDENCE_POLICY_TBD`).
