# PellaV2 Audio Deepfake Detector — Model Card

## Identity
- **Model name:** PellaV2 Audio Deepfake Detector
- **Variant:** Original FP32 (`pellav2_detector.pt`)
- **Upstream Source:** Hugging Face `Sadanie/pellav2-audio-deepfake-detector`
- **Revision:** `94b899915ee551fa90d8844961532e4aaab58322`
- **Backbone:** `facebook/w2v-bert-2.0` (300M parameter Conformer)
- **Multi-layer Pooling:** Learned Softmax Layer Weighting
- **Classification Head:** Linear projection (1024 -> 1)
- **Feature Extraction:** 80-bin Mel Filterbank x 2 frames -> 160-dim log-mel (Kaldi compliance)
- **Decision Boundary:** Sigmoid probability $\ge 0.50 \implies \text{FAKE}$, $< 0.50 \implies \text{GENUINE}$

## Checkpoint Information
- **File size:** 2.32 GB (2,492,059,203 bytes)
- **SHA-256:** `d21aedfb52e2f87706c2753715a29665482b0f4a034800c31579df5d458559ac`
- **Integrity Status:** VERIFIED

## Validated Empirical Performance
- **120-Sample Dataset Benchmark:**
  - Accuracy: 71.67%
  - Precision: 80.95%
  - Recall: 56.67%
  - F1-Score: 0.6667
  - ROC-AUC: 0.8483
  - False Positive Rate: 13.33%
  - False Negative Rate: 43.33%
- **240-Sample Expanded Evaluation:**
  - Accuracy: 66.67%
  - Precision: 73.81%
  - Recall: 51.67%
  - F1-Score: 0.6078
  - ROC-AUC: 0.7557
  - False Positive Rate: 18.33%
  - False Negative Rate: 48.33%
- **24-Sample Live Docker Container Spot Check:**
  - Accuracy: 70.83%
  - Precision: 85.71%
  - Recall: 50.00%
  - F1-Score: 0.6316
  - False Positive Rate: 8.33%
  - False Negative Rate: 50.00%

## Known Limitations
1. **Acoustic Background Masking:** Ambient or synthetic background noise fills vocoder phase/spectral notches, causing ~45–50% false negative rates on noisy synthetic audio.
2. **Multi-Modal Mandate:** The model is not designed for standalone deployment. It must be paired with conversational semantic risk assessment (RIAS) and fused via the complete 15-combination TCED policy.
