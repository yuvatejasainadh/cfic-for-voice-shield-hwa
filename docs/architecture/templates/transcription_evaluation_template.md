# Multilingual Speech Transcription Evaluation Report

**Author:** Pallavi (Cloud Engineer)  
**Date:** YYYY-MM-DD  
**Status:** DRAFT / COMPLETED

---

## 1. Candidate Models Evaluated

1. **Whisper Multilingual:** [Specify model size, e.g. `whisper-small` / `faster-whisper-base`]
2. **AI4Bharat IndicConformer:** [Specify model revision and architecture]

---

## 2. Evaluation Dataset & Test Sets

- **Target Languages:** English, Telugu, Tamil, Kannada, Malayalam
- **Audio Conditions Tested:**
  - Clean studio speech
  - Telephony audio (8 kHz / 16 kHz bandpass)
  - Background noise / street noise
  - Code-switched speech (e.g. Telugu-English, Tamil-English)

---

## 3. Comparative Benchmark Results

| Language | Model | Word Error Rate (WER %) | Char Error Rate (CER %) | P50 Latency (10s audio, ms) | RAM Footprint (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **English** | Whisper | | | | |
| | IndicConformer | | | | |
| **Telugu** | Whisper | | | | |
| | IndicConformer | | | | |
| **Tamil** | Whisper | | | | |
| | IndicConformer | | | | |
| **Kannada** | Whisper | | | | |
| | IndicConformer | | | | |
| **Malayalam** | Whisper | | | | |
| | IndicConformer | | | | |

---

## 4. Final Recommendation & Rationale

- **Recommended Model:** `[Model Name]`
- **Technical Rationale:** [Summarize latency, accuracy, dependency size, and containerization tradeoffs]
- **Known Limitations:** [Document any specific audio lengths or dialect limitations]
