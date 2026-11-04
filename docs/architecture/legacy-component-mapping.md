# CFIC × VoiceShield — Legacy Component Mapping & Deprecation Register

This document tracks the transition from the prototype monolithic scaffold to the HYBRID Architecture (API Gateway + Cloud Inference).

## 1. Overview of Architectural Evolution

In the previous prototype (HackVersion v0.1):
- A monolithic API gateway directly orchestrated PellaV2 (acoustic deepfake detector), a mock transcription service, and a hosted AskDrip LLM service (RIAS).
- TCED fused PellaV2 scores and RIAS risk levels with locked rules.
- Language identification (LID), regional ASR routing, and VMC-South/AASIST models were excluded or marked TBD.

In the approved **HYBRID Architecture**:
- **API Gateway (`api-gateway/`)**: Manages client auth, rate limiting, request validation, audio upload/storage, 5-second chunking, job management, application-level risk aggregation, decision engine, and API responses. Completely decoupled from ML libraries.
- **Cloud Inference (`cloud-inference/`)**: Dedicated subsystem hosting Model Serving, Language Identification (LID), IndicConformer ASR (English, Telugu, Hindi), VMC-South (fine-tuned AASIST-L), and AASIST-L fallback. Controls model routing, preprocessing, inference latency, and result normalization.
- **Legacy Area (`legacy/`)**: Safely preserves earlier contributions by Team SAIL members without polluting active microservices.

---

## 2. Component Disposition Matrix

| Old Component | Original Location | New Disposition | Status | Team Owner | Notes & Dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PellaV2 Detector Service** | `services/pella/` | `legacy/services/pella/` | **Archived** (Replaced by VMC-South & AASIST fallback in Cloud Inference) | Varshu & Varun | Scaffold implementation preserved. Do NOT map directly into VMC-South or AASIST-L without model verification. |
| **PellaV2 Client Stub** | `services/pella/client.py` | `legacy/services/pella/client.py` | **Archived** | Varshu | Replaced by `api-gateway/app/services/cfic_client.py` communicating with Cloud Inference. |
| **Transcription Service Scaffold** | `services/transcription/` | `legacy/services/transcription/` | **Archived** (Replaced by `cloud-inference/models/asr/indicconformer/`) | Pallavi | Earlier mock service archived. Real IndicConformer checkpoints (English, Telugu, Hindi) live under Cloud Inference. |
| **Transcription Client Stub** | `services/transcription/client.py` | `legacy/services/transcription/client.py` | **Archived** | Pallavi | Replaced by `cfic_client.py` inside API Gateway. |
| **RIAS Stage 2 Assessor** | `services/rias/assessor.py`, `prompt.py` | `legacy/services/rias/` & `api-gateway/app/services/risk_service.py` | **Refactored** | Sainadh | Direct hosted LLM client logic migrated to Gateway `risk_service.py` for application-level conversational analysis. Model serving does not touch RIAS. |
| **TCED Decision Policy** | `app/tced/policy.py` | `legacy/tced/` & `api-gateway/app/orchestration/decision_engine.py` | **Refactored** | Sainadh | Hardened safety rules (`RIAS HIGH -> SUSPICIOUS`, fail-safe `INCONCLUSIVE`) encapsulated into `decision_engine.py` in API Gateway. |
| **Prototype API Scaffold** | `app/` | `api-gateway/app/` | **Migrated** | Sainadh | Restructured into modular clean architecture (routes, core, middleware, db, schemas, services, orchestration). |

---

## 3. Team-Owned Code Preservation & Verification Checklist

1. **Varun (MLOps Architect):**
   - Original responsibility: PellaV2 checkpoint verification & quantization.
   - New direction: Prepare VMC-South (fine-tuned AASIST-L for South Indian speech) and AASIST-L fallback weights under `cloud-inference/models/vmc-south/weights/` and `cloud-inference/models/aasist-fallback/weights/`.
   - Legacy code safely retained at `legacy/services/pella/`.

2. **Varshu (Cloud Engineer):**
   - Original responsibility: PellaV2 Docker containerization.
   - New direction: Containerize VMC-South and AASIST-L fallback microservices in `cloud-inference/models/`.
   - Legacy Dockerfile retained at `legacy/services/pella/Dockerfile`.

3. **Pallavi (Cloud Engineer):**
   - Original responsibility: Speech-to-Text benchmark & transcription service.
   - New direction: Containerize IndicConformer English, Telugu, and Hindi models in `cloud-inference/models/asr/indicconformer/`.
   - Legacy benchmark work preserved in `cfic-asr-benchmark/` and service scaffold in `legacy/services/transcription/`.

4. **Sainadh (Project Lead & System Architect):**
   - API Gateway modularization, Cloud Inference orchestrator routing, decision engine, and CI/CD Docker Compose orchestration.
