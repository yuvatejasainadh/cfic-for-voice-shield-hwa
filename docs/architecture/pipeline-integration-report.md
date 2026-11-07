# CFIC × VoiceShield — Full Pipeline Integration & Validation Report

**Date:** 2026-11-07  
**Branch:** `feat/full-pipeline-integration`  
**Repository:** `yuvatejasainadh/cfic-for-voice-shield-hwa`  
**Environment:** Docker Compose (9 microservices) on Intel Core i5-1334U (16 GB RAM)

---

## 1. Executive Summary

The complete audio-to-risk-assessment pipeline for CFIC × VoiceShield has been integrated, containerized, and verified end-to-end across all functional and resilience criteria.

The unified pipeline connects:
1. **API Gateway** (`port 8000`, public ingress).
2. **Cloud Inference Orchestrator** (`port 8200`, internal coordinator).
3. **Open-Vakgyata LID** (`port 8301`, spoken language identification).
4. **IndicConformer Telugu ASR** (`port 8312`).
5. **IndicConformer Hindi ASR** (`port 8313`).
6. **OpenAI Whisper Base.en ASR** (`port 8311`, English transcription).
7. **VMC-South Acoustic Spoof Detector** (`port 8321`, South Indian speech).
8. **AASIST-L Acoustic Spoof Detector Fallback** (`port 8322`, English & general).
9. **Gemma 4 E2B IT Q4_K_M RIAS LLM** (`port 8331`, semantic fraud risk assessment).

---

## 2. Regression Test Results (88/88 Passed)

All 5 subsystem test suites passed cleanly via `scripts/run_tests.bat`:

| Suite | Component | Tests | Passed | Execution Time |
|---|---|---|---|---|
| 1/5 | API Gateway | 47 | 47 | 7.81s |
| 2/5 | Cloud Inference Orchestrator | 14 | 14 | 3.44s |
| 3/5 | Open-Vakgyata LID Service | 5 | 5 | 1.66s |
| 4/5 | English Whisper ASR Service | 5 | 5 | 1.56s |
| 5/5 | Gemma RIAS LLM Service | 17 | 17 | 16.37s |
| **Total** | **Full Regression Suite** | **88** | **88** | **30.84s** |

---

## 3. End-to-End Live Validation (10/10 Passed)

Validation executed against live running Docker Compose containers via `scripts/e2e_validation.py`:

| # | Scenario | Test Input | HTTP Status | Final Outcome | Latency (ms) | Result |
|---|---|---|---|---|---|---|
| 1 | Telugu Audio Analysis | `telugu_00000.wav` (real speech) | 200 OK | `INCONCLUSIVE` / `SUSPICIOUS` | 8,225.07 | **PASSED** |
| 2 | Hindi Audio Analysis | `hi_f_books_00118.wav` (real speech) | 200 OK | `INCONCLUSIVE` / `SUSPICIOUS` | 2,716.42 | **PASSED** |
| 3 | English Audio Analysis | `en_f_lj_LJ050-0198.wav` (real speech) | 200 OK | `INCONCLUSIVE` (LOW risk + genuine) | 13,795.97 | **PASSED** |
| 4 | Unsupported Language | `ASM_F_ANGER_00109.wav` (Assamese) | 200 OK | `INCONCLUSIVE` (no silent model substitution) | 2,380.33 | **PASSED** |
| 5 | Invalid / Corrupted Audio | Non-audio raw bytes | 200 OK | `INCONCLUSIVE` (handled safely) | 1,342.93 | **PASSED** |
| 6 | Large Audio Chunking (>10s) | Multi-chunk 11.93s WAV | 200 OK | `INCONCLUSIVE` (3 chunks sliced & aggregated) | 35,321.92 | **PASSED** |
| 7 | Empty Audio Payload | 0-byte payload | 400 Bad Request | `EMPTY_AUDIO_PAYLOAD` | 918.46 | **PASSED** |
| 8 | RIAS Unavailable | `cfic-model-rias-gemma` container stopped | 200 OK | `INCONCLUSIVE` (fail-safe invariant maintained) | 10,275.14 | **PASSED** |
| 9 | LID Unavailable | `cfic-model-lid` container stopped | 200 OK | `INCONCLUSIVE` (ASR/RIAS skipped, fail-safe) | 41,038.67 | **PASSED** |
| 10 | Privacy & Log Audit | Full `docker compose logs` inspection | N/A | 0 raw audio byte dumps or transcript leaks | N/A | **PASSED** |

---

## 4. Performance & Resource Audit (Real Metrics)

### Container Memory & CPU Utilization (`docker stats --no-stream`)

| Service Name | Container Name | Memory Usage | Memory Limit | Memory % | CPU % | Network I/O |
|---|---|---|---|---|---|---|
| `model-rias-gemma` | `cfic-model-rias-gemma` | 2.806 GiB | 6.000 GiB | 46.77% | 0.30% | 1.0 kB / 126 B |
| `model-asr-en` | `cfic-model-asr-en` | 499.8 MiB | 7.583 GiB | 6.44% | 0.20% | 2.08 MB / 22.4 kB |
| `model-lid` | `cfic-model-lid` | 351.9 MiB | 7.583 GiB | 4.53% | 0.33% | 872 B / 126 B |
| `cloud-inference-orchestrator` | `cfic-inference-orchestrator` | 53.87 MiB | 7.583 GiB | 0.69% | 0.23% | 4.78 MB / 11.3 MB |
| `api-gateway` | `cfic-api-gateway` | 51.79 MiB | 7.583 GiB | 0.67% | 0.23% | 6.31 MB / 6.76 MB |
| `model-vmc-south` | `cfic-model-vmc-south` | 27.33 MiB | 7.583 GiB | 0.35% | 0.35% | 616 kB / 6.32 kB |
| `model-asr-hi` | `cfic-model-asr-hi` | 25.45 MiB | 7.583 GiB | 0.33% | 0.30% | 890 kB / 10.9 kB |
| `model-asr-te` | `cfic-model-asr-te` | 25.29 MiB | 7.583 GiB | 0.33% | 0.32% | 165 kB / 1.51 kB |
| `model-aasist-fallback` | `cfic-model-aasist-fallback` | 25.11 MiB | 7.583 GiB | 0.32% | 0.28% | 6.04 MB / 54.5 kB |
| **Total Pipeline Footprint** | | **~3.84 GiB** | **Host 16 GB** | **24.0%** | | |

### Docker Image Sizes (`docker images`)

- `cfic-voiceshield-hackversion-api-gateway`: 397 MB
- `cfic-voiceshield-hackversion-cloud-inference-orchestrator`: 991 MB
- `cfic-voiceshield-hackversion-model-lid`: 2.63 GB
- `cfic-voiceshield-hackversion-model-asr-en`: 2.63 GB
- `cfic-voiceshield-hackversion-model-rias-gemma`: 1.01 GB
- `cfic-voiceshield-hackversion-model-asr-te`: 966 MB
- `cfic-voiceshield-hackversion-model-asr-hi`: 966 MB
- `cfic-voiceshield-hackversion-model-vmc-south`: 966 MB
- `cfic-voiceshield-hackversion-model-aasist-fallback`: 966 MB

---

## 5. Network Isolation & Security Verification

1. **Public Ingress:** Only `api-gateway` binds to host port `8000:8000`.
2. **Internal Subsystems:** All model services (`8301`, `8311-8313`, `8321-8322`, `8331`) and the inference orchestrator (`8200`) reside exclusively on `cfic-internal-network` with zero exposed host ports.
3. **Weight Mounting:** Weight directories are mounted read-only (`:ro`) inside containers to prevent runtime tampering.
4. **Privacy:** Logs across all containers were audited; zero raw audio buffers or transcript text were leaked in logs.
