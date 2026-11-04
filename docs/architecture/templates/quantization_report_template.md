# PellaV2 Quantization & Verification Report

**Author:** Varun (MLOps Architect)  
**Date:** YYYY-MM-DD  
**Status:** DRAFT / COMPLETED

---

## 1. Artifact Verification

- **Artifact Name / Revision:** [Fill In]
- **Source Repository / URL:** [Fill In]
- **Original Checkpoint Checksum (SHA-256):** `[Fill In]`
- **XLS-R Backbone Embedded:** YES / NO (specify)
- **Native Class Output Mapping:** `0 = [Label]`, `1 = [Label]`
- **Native Score Polarity:** Higher score indicates `[FAKE / GENUINE]`

---

## 2. Quantization Methodology

- **Technique:** [e.g. INT8 Dynamic Post-Training Quantization / FP16 Mixed Precision]
- **Target Runtime:** ONNX Runtime CPU / PyTorch CPU
- **Quantization Script Location:** `scripts/quantize_pellav2.py`

---

## 3. Benchmark Comparison

| Metric | Original FP32 | Quantized (INT8/FP16) | Delta / Change |
| :--- | :--- | :--- | :--- |
| **Model Size (MB)** | | | % |
| **Peak RAM Usage (MB)** | | | % |
| **Cold Start Latency (ms)** | | | ms |
| **P50 Latency (5s audio, ms)** | | | ms |
| **P95 Latency (5s audio, ms)** | | | ms |
| **Equal Error Rate (EER %)** | | | % |
| **Detection Accuracy (%)** | | | % |

---

## 4. Handoff Checklist for Varshu

- [ ] Model weights exported to approved registry / secure storage: `[Storage URI]`
- [ ] SHA-256 checksum validated
- [ ] Inference snippet provided and verified
- [ ] Dependencies documented for `services/pella/requirements.txt`
