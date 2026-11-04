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

1. **Mandatory TCED Rule:**
   - RIAS `HIGH` produces `SUSPICIOUS` even when Pella reports `GENUINE`.
   - RIAS `HIGH` produces `SUSPICIOUS` when Pella reports `FAKE`.
   - RIAS `HIGH` produces `SUSPICIOUS` when Pella fails.
2. **Failure Handling:**
   - Both paths fail -> `INCONCLUSIVE` (reason: `INSUFFICIENT_EVIDENCE`).
   - One path fails -> `INCONCLUSIVE` (reason: `PARTIAL_EVIDENCE_POLICY_TBD`).
   - Unrecognized RIAS risk level -> `INCONCLUSIVE` (never defaults to `LOW`).
3. **Gateway Protection:**
   - Missing or invalid `X-API-Key` returns `401 Unauthorized`.
   - Empty audio payload returns `400 Bad Request`.
   - Payload exceeding 25 MiB returns `413 Payload Too Large`.
   - Exceeding request rate limit returns `429 Too Many Requests` with `Retry-After`.
   - Tracing header `X-Request-ID` is preserved or auto-generated.
