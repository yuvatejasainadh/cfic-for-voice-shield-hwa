# CFIC × VoiceShield HackVersion — Decisions and TBD Register

This document maintains an explicit, traceable record of resolved architectural decisions and open technical items (TBDs) requiring validation.

---

## 1. Resolved Architectural Decisions

| ID | Decision | Resolution | Status |
| :--- | :--- | :--- | :--- |
| **AD-01** | **Primary Deepfake Detector** | Single primary detector model: **PellaV2**. | Locked |
| **AD-02** | **Conversational Risk Path (RIAS)** | Two-stage: Audio transcription followed by hosted AskDrip LLM API risk analysis. | Locked |
| **AD-03** | **No Local LLM** | Do not deploy local LLM instances. The hosted AskDrip LLM API handles RIAS risk analysis. | Locked |
| **AD-04** | **Mandatory TCED Rule** | If RIAS returns `HIGH`, final outcome MUST be `SUSPICIOUS`, regardless of PellaV2 detector output. | Locked |
| **AD-05** | **Failure Fallback Policy** | Missing evidence, upstream timeouts, or service errors return `INCONCLUSIVE`. Never silently classify as `GENUINE`. | Locked |
| **AD-06** | **Excluded Legacy Components** | Formally removed: Separate LID model, LACR routing, 7 language-cluster detectors, DSR, standalone XLS-R service. | Locked |
| **AD-07** | **API Gateway Security** | `X-API-Key` authentication, 25 MiB payload cap, in-memory sliding window rate limiting, and `X-Request-ID` tracing. | Locked |

---

## 2. Open Technical Decisions & TBDs

The following decisions require empirical verification by assigned team members before production deployment:

### TBD-01: Exact PellaV2 Model Artifact & Score Polarity
- **Owner:** Varun (MLOps Architect)
- **Problem:** Need to verify whether the checkpoint is PyTorch (`.pt`) or ONNX, whether XLS-R is embedded, and the exact polarity of the acoustic score (e.g. is score $> 0.5$ fake or genuine?).
- **Action:** Inspect checkpoint, test with known genuine/synthetic samples, and complete `models/artifact_manifest.json`.

### TBD-02: PellaV2 Quantization Strategy
- **Owner:** Varun & Varshu
- **Problem:** Determine whether INT8 dynamic quantization or FP16 produces acceptable Equal Error Rate (EER) trade-offs while meeting the $< 50$ ms latency target.
- **Action:** Execute quantization benchmark and document in `docs/benchmarks/pellav2_quantization_report.md`.

### TBD-03: Multilingual Transcription Model Selection
- **Owner:** Pallavi (Cloud Engineer)
- **Problem:** Whisper Multilingual vs AI4Bharat IndicConformer for target Indian languages (English, Telugu, Tamil, Kannada, Malayalam).
- **Action:** Benchmark WER, inference latency, memory footprint, and code-switching resilience. Document recommendation in `docs/benchmarks/transcription_evaluation_report.md`.

### TBD-04: TCED Hybrid Weights & Score Calibration
- **Owner:** Sainadh (Project Lead) in collaboration with Varun
- **Problem:** Exact mathematical weights between PellaV2 score and RIAS conversational risk for LOW and MEDIUM cases.
- **Action:** Run joint evaluation on synthetic call dataset to calibrate thresholds before enabling `TCED_CALIBRATED = True`.

### TBD-05: Hosted AskDrip LLM Prompt & Threshold Calibration
- **Owner:** Sainadh (Project Lead)
- **Problem:** Refine system prompt instructions for ambiguous borderline calls and calibrate temperature/context window.
- **Action:** Test prompt on representative fraud scripts and confirm JSON parsing reliability.

### TBD-06: Shared Rate Limiting for Clustered Deployment
- **Owner:** Sainadh & Varshu
- **Problem:** Process-local rate limiter is suitable for single instance, but multi-instance cloud deployments require shared storage.
- **Action:** Add Redis-backed sliding window limiter if multiple CFIC API replicas are launched behind a load balancer.
