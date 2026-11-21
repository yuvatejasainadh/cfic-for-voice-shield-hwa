# CFIC × VoiceShield HackVersion — Decisions and TBD Register

This document maintains an explicit, traceable record of resolved architectural decisions and open technical items (TBDs) requiring validation.

---

## 1. Resolved Architectural Decisions

| ID | Decision | Resolution | Status |
| :--- | :--- | :--- | :--- |
| **AD-01** | **Primary Deepfake Detector** | Single primary detector model: **PellaV2**. | Locked |
| **AD-02** | **Conversational Risk Path (RIAS)** | Two-stage: Audio transcription followed by hosted AskDrip LLM API risk analysis. | Locked |
| **AD-03** | **No Local LLM** | Do not deploy local LLM instances. The hosted AskDrip LLM API handles RIAS risk analysis. | Locked |
| **AD-04** | **Mandatory TCED Fusion Policy** | Rules A-H: Acoustic `FAKE` takes precedence (`FAKE`, `ACOUSTIC_SPOOF_DETECTED`), never downgraded by RIAS. If acoustic is `GENUINE`/unavailable and RIAS is `HIGH`, return `SUSPICIOUS` (`HIGH_RIAS_OVERRIDE`). Acoustic unavailable + RIAS `LOW` returns `INCONCLUSIVE` (`ACOUSTIC_EVIDENCE_UNAVAILABLE`). | Locked |
| **AD-05** | **Failure Fallback Policy** | Missing evidence, upstream timeouts, or service errors return `INCONCLUSIVE`. Never silently classify as `GENUINE`. | Locked |
| **AD-06** | **Excluded Legacy Components** | Formally removed: Separate LID model, LACR routing, 7 language-cluster detectors, DSR, standalone XLS-R service. | Locked |
| **AD-07** | **API Gateway Security** | `X-API-Key` authentication, 25 MiB payload cap, in-memory sliding window rate limiting, and `X-Request-ID` tracing. | Locked |

---

## 2. Open Technical Decisions & TBDs

The following decisions require empirical verification by assigned team members before production deployment:

### TBD-01: Exact PellaV2 Model Artifact & Score Polarity [RESOLVED]
- **Owner:** Varun (MLOps Architect) in collaboration with Engineering
- **Resolution:** Checkpoint is PyTorch FP32 (`weights/pellav2_detector.pt`, 2.32 GB, SHA-256 `d21aedfb...`) sourced from `Sadanie/pellav2-audio-deepfake-detector` (revision `94b89991...`). Architecture is `facebook/w2v-bert-2.0` (300M parameter Conformer) with learned layer weights and linear projection head. Output is standard logistic sigmoid score $\in [0, 1]$, where score $\ge 0.50$ indicates `FAKE` and $< 0.50$ indicates `GENUINE`. Status: Locked.

### TBD-02: PellaV2 Quantization Strategy [RESOLVED]
- **Owner:** Varun & Varshu
- **Resolution:** Empirical evaluation of INT8 dynamic quantization revealed severe model degradation (accuracy 47.50%, ROC-AUC 0.5099, false positive rate 58.33%). The original FP32 non-quantized model was integrated and benchmarked under identical conditions, achieving 71.67% accuracy, ROC-AUC 0.8483, and FPR 13.33% (93.33% accuracy on real speech). Decision: Retain original FP32 model in production with 8GB container RAM limit; post-training INT8 is rejected due to acoustic feature corruption. Status: Locked.

### TBD-03: Multilingual Transcription Model Selection
- **Owner:** Pallavi (Cloud Engineer)
- **Problem:** Whisper Multilingual vs AI4Bharat IndicConformer for target Indian languages (English, Telugu, Tamil, Kannada, Malayalam).
- **Action:** Benchmark WER, inference latency, memory footprint, and code-switching resilience. Document recommendation in `docs/benchmarks/transcription_evaluation_report.md`.

### TBD-04: TCED Hybrid Weights & Score Calibration [RESOLVED]
- **Owner:** Sainadh (Project Lead) in collaboration with Varun
- **Resolution:** Deterministic rule-based fusion locked (Rules A-H). When acoustic detector is unavailable and RIAS is LOW, outcome is `INCONCLUSIVE` with reason `ACOUSTIC_EVIDENCE_UNAVAILABLE`. LOW semantic risk is never a substitute for authentic acoustic speech evidence. Verified against deepfake dataset. Status: Locked.

### TBD-05: Hosted AskDrip LLM Prompt & Threshold Calibration
- **Owner:** Sainadh (Project Lead)
- **Problem:** Refine system prompt instructions for ambiguous borderline calls and calibrate temperature/context window.
- **Action:** Test prompt on representative fraud scripts and confirm JSON parsing reliability.

### TBD-06: Shared Rate Limiting for Clustered Deployment
- **Owner:** Sainadh & Varshu
- **Problem:** Process-local rate limiter is suitable for single instance, but multi-instance cloud deployments require shared storage.
- **Action:** Add Redis-backed sliding window limiter if multiple CFIC API replicas are launched behind a load balancer.
