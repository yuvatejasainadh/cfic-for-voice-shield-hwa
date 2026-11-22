# CFIC × VoiceShield HackVersion — Testing Guide

This document defines the testing strategy, test suites, and instructions for verifying the CFIC platform.

---

## 1. Test Architecture

The test suite is organized into three decoupled layers:

```
tests/
├── unit/
│   ├── test_tced.py             # Pure decision logic and safety overrides
│   ├── test_schemas.py          # Request and response schema validation
│   ├── test_rate_limiter.py     # Sliding window rate limiter enforcement
│   └── test_rias_assessor.py    # LLM prompt construction & response parsing
├── integration/
│   ├── test_api_health.py       # Gateway /health endpoint
│   ├── test_api_analyze.py      # End-to-end /v1/analyze with mocked downstream
│   └── test_api_auth_limits.py  # Auth, payload size, empty file, rate limiting
└── contracts/
    ├── test_pella_contract.py   # PellaV2 service API contract compliance
    └── test_transcription_contract.py # Transcription service API contract compliance
```

---

## 2. Mock vs Real Model Policy

- **Unit & Integration Tests:** Use explicit mocks for external network calls (PellaV2 service, Transcription service, AskDrip LLM API). Tests run fast and completely offline with zero external dependencies or large model weight downloads.
- **Contract Tests:** Validate schema and behavior of HTTP clients against mock server fixtures representing downstream microservices.
- **Reporting Rule:** Never report mock test executions as proof of live deepfake detection or speech recognition accuracy. Live model validations are recorded separately in benchmark reports by Varun and Pallavi.

---

## 3. Running the Test Suite

### 3.1 Run All Tests
```bash
python -m pytest -v
```

### 3.2 Run Specific Test Suites
```bash
# Run unit tests only
python -m pytest tests/unit -v

# Run integration tests
python -m pytest tests/integration -v

# Run contract tests
python -m pytest tests/contracts -v
```

### 3.3 Run TCED Safety Rules Specifically
```bash
python -m pytest tests/unit/test_tced.py -v -k "high_rias"
```

---

## 4. Key Test Scenarios Covered

1. **Complete TCED 15-Combination Policy:**
   - Acoustic `FAKE` produces `FAKE` (`ACOUSTIC_SPOOF_DETECTED`) across all RIAS states (`LOW`, `MEDIUM`, `HIGH`, `UNAVAILABLE`, `INVALID`).
   - Acoustic `GENUINE` + RIAS `LOW` exclusively produces `GENUINE` (`ACOUSTIC_GENUINE_CONFIRMED`).
   - Acoustic `GENUINE` + RIAS `MEDIUM` produces `INCONCLUSIVE` (`EVIDENCE_CONFLICT`).
   - Acoustic `GENUINE` + RIAS `HIGH` produces `SUSPICIOUS` (`HIGH_RIAS_OVERRIDE`).
   - Acoustic `GENUINE` + RIAS `UNAVAILABLE` produces `INCONCLUSIVE` (`RIAS_EVIDENCE_UNAVAILABLE`).
   - Acoustic `GENUINE` + RIAS `INVALID` produces `INCONCLUSIVE` (`INVALID_RIAS_RISK_LEVEL`).
   - Acoustic `UNAVAILABLE` + RIAS `LOW` produces `INCONCLUSIVE` (`ACOUSTIC_EVIDENCE_UNAVAILABLE`).
   - Acoustic `UNAVAILABLE` + RIAS `MEDIUM` produces `INCONCLUSIVE` (`INSUFFICIENT_EVIDENCE`).
   - Acoustic `UNAVAILABLE` + RIAS `HIGH` produces `SUSPICIOUS` (`HIGH_RIAS_OVERRIDE`).
   - Acoustic `UNAVAILABLE` + RIAS `UNAVAILABLE` produces `INCONCLUSIVE` (`INSUFFICIENT_EVIDENCE`).
   - Acoustic `UNAVAILABLE` + RIAS `INVALID` produces `INCONCLUSIVE` (`INVALID_RIAS_RISK_LEVEL`).
2. **Failure Handling & Edge Cases:**
   - Both paths fail -> `INCONCLUSIVE` (reason: `INSUFFICIENT_EVIDENCE`).
   - Missing/unavailable acoustic evidence -> `INCONCLUSIVE` (reason: `ACOUSTIC_EVIDENCE_UNAVAILABLE`).
   - Missing/unavailable RIAS evidence on authentic audio -> `INCONCLUSIVE` (reason: `RIAS_EVIDENCE_UNAVAILABLE`).
   - Unrecognized RIAS risk level -> `INCONCLUSIVE` (reason: `INVALID_RIAS_RISK_LEVEL`, never defaults to `LOW`).
   - None, empty dictionaries, and malformed inputs handled safely without exceptions.
3. **Gateway Protection & Security Hardening:**
   - Missing, empty, or invalid `X-API-Key` returns `401 Unauthorized` (`UNAUTHORIZED`) with constant-time verification.
   - Empty audio payload returns `400 Bad Request` (`EMPTY_AUDIO_PAYLOAD`).
   - Payload exceeding 25 MiB returns `413 Payload Too Large` (`AUDIO_PAYLOAD_TOO_LARGE`).
   - Unsupported MIME types (e.g., `text/plain`) or file extensions (e.g., `.txt`, `.pdf`) return `415 Unsupported Media Type` (`UNSUPPORTED_AUDIO_FORMAT`).
   - Corrupted or unparseable audio bitstreams return `400 Bad Request` (`CORRUPTED_AUDIO_PAYLOAD`).
   - Audio files exceeding maximum duration (120s) return `400 Bad Request` (`AUDIO_DURATION_TOO_LONG`).
   - Exceeding request rate limit returns `429 Too Many Requests` (`RATE_LIMIT_EXCEEDED`) with `Retry-After` header.
   - Inactive rate limit clients are periodically pruned to prevent memory exhaustion under high-volume client rotation.
   - Bounded concurrency and queue manager rejects overload with `503 Service Unavailable` (`SERVICE_OVERLOADED`) when queue is full.
   - Queue wait timeouts abort with `504 Gateway Timeout` (`QUEUE_TIMEOUT`).
   - Tracing header `X-Request-ID` is preserved in both response headers and payload body.

---

## 5. Postman Collection & Automated Validation

The repository includes a comprehensive Postman test suite and environment in the `postman/` directory:
- `postman/CFIC_VoiceShield_API_Gateway_Validation.postman_collection.json`
- `postman/CFIC_VoiceShield_Local.postman_environment.json`

### 5.1 Test Suite Organization (8 Functional Folders, 29 Requests, 63 Assertions)
1. `1. Health and Discovery`: System health, versioned health, Swagger, and ReDoc discovery.
2. `2. Authentication and Authorization`: Missing, invalid, empty, and valid API key scenarios.
3. `3. Rate Limiting`: Normal quota, burst limit 429, and `Retry-After` validation.
4. `4. Request Validation`: Empty (400), oversized (413), invalid MIME (415), corrupt audio (400), long audio (400).
5. `5. Audio Chunking`: Short (<5s), 5s window boundary, and multi-chunk (10s) aggregation.
6. `6. Concurrency and Bounded Queue`: Normal slot acquisition with `queue_wait` latency, queue saturation (503), queue timeout (504).
7. `7. End-to-End Orchestration & TCED`: Complete decision engine policy verification (GENUINE, FAKE, SUSPICIOUS, INCONCLUSIVE).
8. `8. Edge Cases & Resiliency`: Custom `X-Request-ID` echoing, special characters in filenames, binary octet-streams, rapid sequential calls.

### 5.2 Running the Postman Validation Suite
Execute the automated validator script:
```bash
python scripts/validate_postman_collection.py
```
This script audits:
- Postman Collection v2.1.0 schema compliance and environment variable resolution.
- Presence of all 8 functional folders and 63 automated `pm.test` assertions.
- Live HTTP/ASGI execution verifying response status codes and body schemas.
