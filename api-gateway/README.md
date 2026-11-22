# CloudForge Intelligence Core (CFIC) — API Gateway

The **CFIC API Gateway** provides the central ingress, security enforcement, audio chunking, and orchestration boundary for **VoiceShield**. It interfaces between client edge devices / apps and downstream deepfake detection and semantic risk analysis microservices.

---

## 1. Core Architecture & Capabilities

```
+-----------------------------------------------------------------------------------+
|                                 VoiceShield Client                                |
+-----------------------------------------------------------------------------------+
                                          |
                         HTTP POST /v1/analyze (multipart/form-data)
                         Headers: X-API-Key, [X-Request-ID]
                                          v
+-----------------------------------------------------------------------------------+
|                                 CFIC API Gateway                                  |
|                                                                                   |
|  1. Request ID Tracing        (Assigns / echoes UUID in headers & payload)        |
|  2. Security & Auth           (Constant-time secrets.compare_digest verification) |
|  3. Rate Limiting             (Sliding-window bounded memory tracking, 429)       |
|  4. Request Validation        (MIME check, 25MB limit, decodability, 120s cap)    |
|  5. 5-Second Window Chunking  (Sample-conserving sequential window slicing)       |
|  6. Bounded Concurrency Queue (Semaphore slots, bounded queue, 503/504 backpressure)|
+-----------------------------------------------------------------------------------+
           |                                                      |
           v                                                      v
+-----------------------+                              +-----------------------+
|  PellaV2 FP32 Detector|                              |  Gemma RIAS LLM / ASR |
|  (Acoustic Spoof)     |                              |  (Semantic Risk)      |
+-----------------------+                              +-----------------------+
           \                                                      /
            +-------------------------+--------------------------+
                                      |
                                      v
                        +---------------------------+
                        |   TCED Decision Engine    |
                        |   (15-Combination Matrix) |
                        +---------------------------+
                                      |
                                      v
                          Normalized AnalysisResponse
```

---

## 2. Security Hardening

### 2.1 Authentication (`X-API-Key`)
- Every request to `/v1/analyze` requires the `X-API-Key` header.
- Validated using `secrets.compare_digest` in `app/core/security.py` to prevent timing attacks.
- Missing, invalid, or empty keys immediately return `401 Unauthorized` with error code `UNAUTHORIZED`.
- Client key contents are never logged or stored directly in internal tracking; internal rate limiting derives a SHA-256 key fingerprint (`get_api_key_fingerprint`).

### 2.2 Sliding-Window Rate Limiting
- Tracks request timestamps per `client_ip:key_fingerprint` in a sliding time window (default: 30 requests / 60 seconds).
- Bounded memory management: automatically purges inactive client tracking entries every 30 seconds or when table exceeds `max_tracked_clients` (10,000 entries).
- Quota exhaustion returns `429 Too Many Requests` with a mandatory `Retry-After` header indicating seconds until next available quota slot.

### 2.3 Audio Ingestion & Validation
The gateway inspects uploaded audio before dispatching downstream compute:
- **Empty payload:** 0-byte uploads return `400 Bad Request` (`EMPTY_AUDIO_PAYLOAD`).
- **Payload size:** Uploads exceeding `max_audio_bytes` (default: 25 MiB) return `413 Payload Too Large` (`AUDIO_PAYLOAD_TOO_LARGE`).
- **MIME & Extension:** Supports WAV, MP3, OGG, and FLAC. Disallowed MIME types or unsupported file extensions return `415 Unsupported Media Type` (`UNSUPPORTED_AUDIO_FORMAT`).
- **Decodability:** Uploaded audio bitstreams are verified via header metadata (`soundfile` / `wave`). Non-audio, damaged, or unparseable files return `400 Bad Request` (`CORRUPTED_AUDIO_PAYLOAD`).
- **Duration Cap:** Audios exceeding `max_audio_duration_seconds` (default: 120.0s) return `400 Bad Request` (`AUDIO_DURATION_TOO_LONG`).

### 2.4 Bounded Concurrency & Queue Management
- **Inference Concurrency Cap:** Bounded execution slots (`max_concurrent_inferences`, default: 10).
- **Waiting Queue:** Bounded queue capacity (`max_queue_capacity`, default: 20). When saturated, incoming requests immediately receive `503 Service Unavailable` (`SERVICE_OVERLOADED`) with a `Retry-After: 5` header.
- **Queue Timeout:** If an enqueued request waits longer than `queue_timeout_seconds` (default: 10.0s), the gateway aborts waiting and returns `504 Gateway Timeout` (`QUEUE_TIMEOUT`).
- **Queue Wait Latency:** All successful responses include `queue_wait` latency in `latency_ms` to provide observability into cluster queueing delays.

---

## 3. Audio Chunking (5-Second Windows)

Incoming audio recordings are dynamically segmented by `AudioService.slice_into_5s_chunks`:
- Audio $\le 5.0$ seconds: Processed directly as a single chunk with zero padding distortion.
- Audio $> 5.0$ seconds: Sliced into sequential, non-overlapping 5-second PCM WAV chunks.
- Sample conservation: Guaranteed $100\%$ sample count conservation with zero dropped frames.

---

## 4. Error Code Reference

| Status Code | Error Code | Description |
|:---|:---|:---|
| `400 Bad Request` | `EMPTY_AUDIO_PAYLOAD` | Uploaded audio file is 0 bytes |
| `400 Bad Request` | `CORRUPTED_AUDIO_PAYLOAD` | Audio bitstream cannot be decoded or has 0 duration |
| `400 Bad Request` | `AUDIO_DURATION_TOO_LONG` | Audio exceeds max duration (120s) |
| `401 Unauthorized` | `UNAUTHORIZED` | Missing, empty, or invalid `X-API-Key` |
| `413 Payload Too Large` | `AUDIO_PAYLOAD_TOO_LARGE` | Upload exceeds max byte size (25MB) |
| `415 Unsupported Media Type` | `UNSUPPORTED_AUDIO_FORMAT` | MIME type or file extension is not WAV/MP3/OGG/FLAC |
| `429 Too Many Requests` | `RATE_LIMIT_EXCEEDED` | Request frequency exceeds sliding-window quota |
| `503 Service Unavailable` | `SERVICE_OVERLOADED` | Inference execution slots and queue are at capacity |
| `504 Gateway Timeout` | `QUEUE_TIMEOUT` | Request timed out while waiting in inference queue |

---

## 5. Postman Collection & Validation

A complete Postman test collection and environment are provided in the `postman/` directory:
- `postman/CFIC_VoiceShield_API_Gateway_Validation.postman_collection.json`
- `postman/CFIC_VoiceShield_Local.postman_environment.json`

### 5.1 Structure (8 Folders, 29 Requests, 63 Assertions)
1. `1. Health and Discovery`: `/health`, `/v1/health`, `/docs`, `/redoc`
2. `2. Authentication and Authorization`: missing key, invalid key, empty key, valid key
3. `3. Rate Limiting`: normal request, burst limit 429, `Retry-After` header check
4. `4. Request Validation`: empty file 400, oversized 413, invalid MIME 415, corrupt audio 400, long audio 400
5. `5. Audio Chunking`: short sample, 5s boundary, 10s multi-chunk
6. `6. Concurrency and Bounded Queue`: slot acquisition, queue saturation 503, queue timeout 504
7. `7. End-to-End Orchestration & TCED`: GENUINE, FAKE, SUSPICIOUS (override), INCONCLUSIVE
8. `8. Edge Cases & Resiliency`: custom `X-Request-ID` echoing, special characters in filenames, binary octet-streams, rapid repeated calls

### 5.2 Automated Validation Script
Run the automated Postman validation suite locally:
```bash
python scripts/validate_postman_collection.py
```
This script verifies:
1. JSON schema v2.1.0 and environment compliance.
2. Structure and presence of all 8 required functional folders.
3. JavaScript `pm.test` assertion coverage across all requests.
4. Executes real/simulated HTTP requests through the gateway and verifies all response codes and payload structures.
