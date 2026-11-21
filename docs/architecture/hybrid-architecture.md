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

## 3. Decision Engine & Locked Safety Rules (TCED Complete Policy)

The Decision Engine (`api-gateway/app/orchestration/decision_engine.py`) enforces strict safety guarantees across an authoritative 15-combination matrix:

### Authoritative 15-Combination Decision Matrix

| Acoustic | RIAS | Final Outcome | Reason Code |
| :--- | :--- | :--- | :--- |
| **FAKE** | **LOW** | `FAKE` | `ACOUSTIC_SPOOF_DETECTED` |
| **FAKE** | **MEDIUM** | `FAKE` | `ACOUSTIC_SPOOF_DETECTED` |
| **FAKE** | **HIGH** | `FAKE` | `ACOUSTIC_SPOOF_DETECTED` |
| **FAKE** | **UNAVAILABLE** | `FAKE` | `ACOUSTIC_SPOOF_DETECTED` |
| **FAKE** | **INVALID** | `FAKE` | `ACOUSTIC_SPOOF_DETECTED` |
| **GENUINE** | **LOW** | `GENUINE` | `ACOUSTIC_GENUINE_CONFIRMED` |
| **GENUINE** | **MEDIUM** | `INCONCLUSIVE` | `EVIDENCE_CONFLICT` |
| **GENUINE** | **HIGH** | `SUSPICIOUS` | `HIGH_RIAS_OVERRIDE` |
| **GENUINE** | **UNAVAILABLE** | `INCONCLUSIVE` | `RIAS_EVIDENCE_UNAVAILABLE` |
| **GENUINE** | **INVALID** | `INCONCLUSIVE` | `INVALID_RIAS_RISK_LEVEL` |
| **UNAVAILABLE** | **LOW** | `INCONCLUSIVE` | `ACOUSTIC_EVIDENCE_UNAVAILABLE` |
| **UNAVAILABLE** | **MEDIUM** | `INCONCLUSIVE` | `INSUFFICIENT_EVIDENCE` |
| **UNAVAILABLE** | **HIGH** | `SUSPICIOUS` | `HIGH_RIAS_OVERRIDE` |
| **UNAVAILABLE** | **UNAVAILABLE** | `INCONCLUSIVE` | `INSUFFICIENT_EVIDENCE` |
| **UNAVAILABLE** | **INVALID** | `INCONCLUSIVE` | `INVALID_RIAS_RISK_LEVEL` |

### Core Decision Priorities
1. **Priority 1 (Acoustic Spoof Priority):** Acoustic `FAKE` takes absolute precedence (`FAKE`, `ACOUSTIC_SPOOF_DETECTED`) over all RIAS states (`LOW`, `MEDIUM`, `HIGH`, `UNAVAILABLE`, `INVALID`). Acoustic spoof evidence is never downgraded.
2. **Priority 2 (High Risk Override):** When acoustic detection returns `GENUINE` or `UNAVAILABLE`, but RIAS returns `HIGH`, the final outcome is `SUSPICIOUS` (`HIGH_RIAS_OVERRIDE`).
3. **Priority 3 (Invalid Risk Guard):** When acoustic is not `FAKE` and RIAS returns an invalid or unrecognized risk level, the outcome is `INCONCLUSIVE` (`INVALID_RIAS_RISK_LEVEL`). Malformed data is never silently treated as low risk.
4. **Priority 4 (Confirmed GENUINE):** `GENUINE` is exclusively produced by `Acoustic GENUINE + RIAS LOW` (`ACOUSTIC_GENUINE_CONFIRMED`).
5. **Priority 5 (Remaining Evidence Handling):**
   - `GENUINE + MEDIUM` returns `INCONCLUSIVE` (`EVIDENCE_CONFLICT`). Medium risk conversational anomalies preclude declaring authentic genuine.
   - `GENUINE + UNAVAILABLE` returns `INCONCLUSIVE` (`RIAS_EVIDENCE_UNAVAILABLE`). Missing semantic risk prevents silent genuine pass-through.
   - `UNAVAILABLE + LOW` returns `INCONCLUSIVE` (`ACOUSTIC_EVIDENCE_UNAVAILABLE`). Low semantic risk is not proof of authentic speech.
   - `UNAVAILABLE + MEDIUM/UNAVAILABLE` returns `INCONCLUSIVE` (`INSUFFICIENT_EVIDENCE`). Missing evidence cannot prove authenticity.
