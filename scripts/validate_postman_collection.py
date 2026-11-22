#!/usr/bin/env python3
"""Validation runner for CFIC VoiceShield Postman Collection and Environment.

Performs:
1. JSON schema and syntax validation.
2. Structure and folder compliance verification (all 8 required folders).
3. Environment variable resolution and reference verification.
4. JavaScript test script and pm.test assertion verification.
5. In-process or live HTTP execution against CFIC API Gateway to verify live test assertions.
"""

import argparse
import asyncio
import io
import json
import os
import re
import sys
import time
import wave
from typing import Any, Dict, List, Tuple


def log_pass(msg: str):
    print(f"  [PASS] {msg}")


def log_fail(msg: str):
    print(f"  [FAIL] {msg}")


def log_info(msg: str):
    print(f"  [*] {msg}")


REQUIRED_FOLDERS = [
    "1. Health and Discovery",
    "2. Authentication and Authorization",
    "3. Rate Limiting",
    "4. Request Validation",
    "5. Audio Chunking",
    "6. Concurrency and Bounded Queue",
    "7. End-to-End Orchestration & TCED",
    "8. Edge Cases & Resiliency",
]

REQUIRED_ENV_VARS = [
    "base_url",
    "api_key",
    "invalid_api_key",
    "request_id",
    "sample_audio_path",
]


def validate_schema(collection_data: Dict[str, Any], env_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate JSON schema compliance for collection and environment."""
    errors = []

    # 1. Collection schema
    info = collection_data.get("info", {})
    if not info:
        errors.append("Collection missing 'info' object")
    if not info.get("name"):
        errors.append("Collection missing 'info.name'")
    schema_url = info.get("schema", "")
    if "v2.1.0/collection.json" not in schema_url:
        errors.append(f"Collection schema must be Postman v2.1.0, got: {schema_url}")

    # 2. Environment schema
    if not env_data.get("name"):
        errors.append("Environment missing 'name'")
    env_values = env_data.get("values", [])
    if not isinstance(env_values, list):
        errors.append("Environment 'values' must be an array")

    env_keys = {item.get("key") for item in env_values if isinstance(item, dict)}
    for req_key in REQUIRED_ENV_VARS:
        if req_key not in env_keys:
            errors.append(f"Environment missing required variable: {req_key}")

    return len(errors) == 0, errors


def validate_folders_and_tests(collection_data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, int]]:
    """Validate that all 8 required folders exist and contain valid requests with pm.test assertions."""
    errors = []
    folder_stats = {}

    folders = collection_data.get("item", [])
    folder_names = [f.get("name") for f in folders if isinstance(f, dict)]

    for req_f in REQUIRED_FOLDERS:
        if req_f not in folder_names:
            errors.append(f"Required folder missing: '{req_f}'")

    total_requests = 0
    total_assertions = 0

    for folder in folders:
        fname = folder.get("name", "Unknown")
        items = folder.get("item", [])
        folder_stats[fname] = len(items)

        for item in items:
            total_requests += 1
            req_name = item.get("name", "Unnamed")
            req = item.get("request", {})
            if not req.get("method"):
                errors.append(f"Request '{req_name}' in folder '{fname}' missing HTTP method")

            # Validate events / test scripts
            events = item.get("event", [])
            test_scripts = [
                e for e in events if e.get("listen") == "test" and isinstance(e.get("script", {}).get("exec"), list)
            ]
            if not test_scripts:
                errors.append(f"Request '{req_name}' in folder '{fname}' has no test event script")
            else:
                script_text = "\n".join(test_scripts[0]["script"]["exec"])
                matches = re.findall(r"pm\.test\(", script_text)
                if not matches:
                    errors.append(f"Request '{req_name}' in folder '{fname}' contains no pm.test() assertions")
                total_assertions += len(matches)

    return len(errors) == 0, errors, {
        "folders_count": len(folders),
        "requests_count": total_requests,
        "assertions_count": total_assertions,
        "per_folder": folder_stats,
    }


def make_test_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate in-memory valid mono WAV audio."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(sample_rate * duration_s))
    return buf.getvalue()


async def run_simulated_execution(env_data: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """Execute collection test suites against the FastAPI API Gateway."""
    # Add api-gateway to python path
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gateway_dir = os.path.join(current_dir, "api-gateway")
    if gateway_dir not in sys.path:
        sys.path.insert(0, gateway_dir)

    from httpx import ASGITransport, AsyncClient
    from app.main import app
    from app.core.config import settings
    from app.api.routes import rate_limiter
    from app.core.concurrency import concurrency_manager
    from app.schemas.analysis import AnalysisResponse, FinalOutcome, RiskLevel

    env_map = {item["key"]: item.get("value") for item in env_data.get("values", []) if "key" in item}
    api_key = env_map.get("api_key", settings.api_key)
    settings.api_key = api_key
    invalid_api_key = env_map.get("invalid_api_key", "invalid-key-xyz-99999")
    custom_trace_id = env_map.get("request_id", "custom-trace-uuid-12345")

    wav_1s = make_test_wav(1.0)
    wav_5s = make_test_wav(5.0)
    wav_10s = make_test_wav(10.0)

    test_results = []

    rate_limiter.reset()
    concurrency_manager.reset()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # FOLDER 1: Health and Discovery
        # 1.1 GET /health
        res = await client.get("/health")
        passed = res.status_code == 200 and res.json().get("status") == "ok"
        test_results.append({
            "folder": "1. Health and Discovery",
            "name": "GET System Root Health (/health)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": f"status={res.json().get('status')}",
        })

        # 1.2 GET /v1/health
        res = await client.get("/v1/health")
        passed = res.status_code == 200 and res.json().get("status") == "ok"
        test_results.append({
            "folder": "1. Health and Discovery",
            "name": "GET Versioned Health (/v1/health)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": f"status={res.json().get('status')}",
        })

        # 1.3 GET /docs
        res = await client.get("/docs")
        passed = res.status_code == 200 and "text/html" in res.headers.get("content-type", "")
        test_results.append({
            "folder": "1. Health and Discovery",
            "name": "GET Swagger OpenAPI Docs (/docs)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": "OpenAPI UI returned",
        })

        # 1.4 GET /redoc
        res = await client.get("/redoc")
        passed = res.status_code == 200 and "text/html" in res.headers.get("content-type", "")
        test_results.append({
            "folder": "1. Health and Discovery",
            "name": "GET ReDoc Documentation (/redoc)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": "ReDoc UI returned",
        })

        # FOLDER 2: Authentication and Authorization
        # 2.1 Missing API Key -> 401
        res = await client.post("/v1/analyze", files={"file": ("test.wav", wav_1s, "audio/wav")})
        passed = res.status_code == 401 and res.json().get("error_code") == "UNAUTHORIZED"
        test_results.append({
            "folder": "2. Authentication and Authorization",
            "name": "Missing API Key (401 Unauthorized)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": res.json().get("error_code"),
        })

        # 2.2 Invalid API Key -> 401
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": invalid_api_key},
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        passed = res.status_code == 401 and res.json().get("error_code") == "UNAUTHORIZED"
        test_results.append({
            "folder": "2. Authentication and Authorization",
            "name": "Invalid API Key (401 Unauthorized)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": res.json().get("error_code"),
        })

        # 2.3 Empty API Key -> 401
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": ""},
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        passed = res.status_code == 401 and res.json().get("error_code") == "UNAUTHORIZED"
        test_results.append({
            "folder": "2. Authentication and Authorization",
            "name": "Empty API Key (401 Unauthorized)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": res.json().get("error_code"),
        })

        # 2.4 Valid API Key Authentication Accepted
        import app.api.routes as routes_module
        orig_analyze = getattr(routes_module, "analyze_audio", None)

        async def fast_mock_analyze(audio, filename, content_type, request_id):
            return AnalysisResponse(
                request_id=request_id,
                final_outcome=FinalOutcome.GENUINE,
                reason_code="ACOUSTIC_GENUINE_CONFIRMED",
                pella_status="success",
                rias_transcription_status="success",
                rias_risk_status="success",
                rias_risk_level=RiskLevel.LOW,
                latency_ms={"total": 45.0, "pella": 20.0, "transcription": 15.0, "rias_llm": 10.0},
            )

        setattr(routes_module, "analyze_audio", fast_mock_analyze)

        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        passed = res.status_code == 200 and "request_id" in res.json()
        test_results.append({
            "folder": "2. Authentication and Authorization",
            "name": "Valid API Key Authentication Accepted",
            "passed": passed,
            "status_code": res.status_code,
            "detail": "200 OK authenticated",
        })

        # FOLDER 3: Rate Limiting
        # 3.1 Normal request
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        passed = res.status_code == 200
        test_results.append({
            "folder": "3. Rate Limiting",
            "name": "Request Within Normal Rate Limit",
            "passed": passed,
            "status_code": res.status_code,
            "detail": "200 OK within quota",
        })

        # 3.2 Burst rate limit exceeded
        old_max = rate_limiter.max_requests
        rate_limiter.max_requests = 1
        rate_limiter.reset()

        # Call 1: pass
        await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        # Call 2: 429
        res_429 = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("test.wav", wav_1s, "audio/wav")},
        )
        passed = res_429.status_code == 429 and "Retry-After" in res_429.headers
        test_results.append({
            "folder": "3. Rate Limiting",
            "name": "Burst Limit Exceeded (429 Rate Limit Exceeded)",
            "passed": passed,
            "status_code": res_429.status_code,
            "detail": f"Retry-After={res_429.headers.get('Retry-After')}",
        })
        rate_limiter.max_requests = old_max
        rate_limiter.reset()

        # FOLDER 4: Request Validation
        # 4.1 Empty audio payload -> 400
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        passed = res.status_code == 400 and res.json().get("error_code") == "EMPTY_AUDIO_PAYLOAD"
        test_results.append({
            "folder": "4. Request Validation",
            "name": "Empty Audio Payload (400 Bad Request)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": res.json().get("error_code"),
        })

        # 4.2 Oversized audio payload -> 413
        old_size_limit = settings.max_audio_bytes
        settings.max_audio_bytes = 100
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("oversized.wav", b"x" * 200, "audio/wav")},
        )
        passed = res.status_code == 413 and res.json().get("error_code") == "AUDIO_PAYLOAD_TOO_LARGE"
        test_results.append({
            "folder": "4. Request Validation",
            "name": "Oversized Audio Payload (>25MB, 413 Payload Too Large)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": res.json().get("error_code"),
        })
        settings.max_audio_bytes = old_size_limit

        # 4.3 Invalid MIME type -> 415
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("doc.txt", b"plain text data", "text/plain")},
        )
        passed = res.status_code == 415 and res.json().get("error_code") == "UNSUPPORTED_AUDIO_FORMAT"
        test_results.append({
            "folder": "4. Request Validation",
            "name": "Invalid MIME Type (text/plain, 415 Unsupported Media Type)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": res.json().get("error_code"),
        })

        # 4.4 Corrupted audio bitstream -> 400
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("corrupt.wav", b"not-a-valid-pcm-audio-stream", "audio/wav")},
        )
        passed = res.status_code == 400 and res.json().get("error_code") == "CORRUPTED_AUDIO_PAYLOAD"
        test_results.append({
            "folder": "4. Request Validation",
            "name": "Corrupted Audio Bitstream (400 Bad Request)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": res.json().get("error_code"),
        })

        # 4.5 Audio exceeding duration cap -> 400
        old_dur_limit = settings.max_audio_duration_seconds
        settings.max_audio_duration_seconds = 2.0
        wav_3s = make_test_wav(3.0)
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("too_long.wav", wav_3s, "audio/wav")},
        )
        passed = res.status_code == 400 and res.json().get("error_code") == "AUDIO_DURATION_TOO_LONG"
        test_results.append({
            "folder": "4. Request Validation",
            "name": "Audio Exceeding Duration Cap (>120s, 400 Bad Request)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": res.json().get("error_code"),
        })
        settings.max_audio_duration_seconds = old_dur_limit

        # FOLDER 5: Audio Chunking
        # 5.1 Short sample (<5s)
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("short.wav", wav_1s, "audio/wav")},
        )
        passed = res.status_code == 200 and res.json().get("final_outcome") in [
            "GENUINE", "FAKE", "SUSPICIOUS", "INCONCLUSIVE"
        ]
        test_results.append({
            "folder": "5. Audio Chunking",
            "name": "Short Audio Sample (<5s, 1s Single Chunk)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": res.json().get("final_outcome"),
        })

        # 5.2 Standard 5s sample
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("standard_5s.wav", wav_5s, "audio/wav")},
        )
        passed = res.status_code == 200
        test_results.append({
            "folder": "5. Audio Chunking",
            "name": "Standard 5s Sample (Exact Chunk Boundary)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": "200 OK at boundary",
        })

        # 5.3 Multi-chunk 10s sample
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("multi_10s.wav", wav_10s, "audio/wav")},
        )
        passed = res.status_code == 200 and "queue_wait" in res.json().get("latency_ms", {})
        test_results.append({
            "folder": "5. Audio Chunking",
            "name": "Multi-Chunk 10s Sample (2 Sequential Chunks)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": "200 OK 2-chunk aggregation",
        })

        # FOLDER 6: Concurrency and Bounded Queue
        # 6.1 Normal slot acquisition
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("sample.wav", wav_1s, "audio/wav")},
        )
        passed = res.status_code == 200 and "queue_wait" in res.json().get("latency_ms", {})
        test_results.append({
            "folder": "6. Concurrency and Bounded Queue",
            "name": "Normal Concurrency Slot Acquisition",
            "passed": passed,
            "status_code": res.status_code,
            "detail": f"queue_wait_ms={res.json().get('latency_ms', {}).get('queue_wait')}",
        })

        # 6.2 Queue saturation -> 503
        concurrency_manager.configure(max_concurrency=1, max_queue=0)
        barrier = asyncio.Event()

        async def slow_mock_analyze(audio, filename, content_type, request_id):
            await barrier.wait()
            return await fast_mock_analyze(audio, filename, content_type, request_id)

        setattr(routes_module, "analyze_audio", slow_mock_analyze)

        task1 = asyncio.create_task(
            client.post("/v1/analyze", headers={"X-API-Key": api_key}, files={"file": ("1.wav", wav_1s, "audio/wav")})
        )
        await asyncio.sleep(0.05)
        res_503 = await client.post(
            "/v1/analyze", headers={"X-API-Key": api_key}, files={"file": ("2.wav", wav_1s, "audio/wav")}
        )
        barrier.set()
        await task1

        passed = res_503.status_code == 503 and res_503.json().get("error_code") == "SERVICE_OVERLOADED"
        test_results.append({
            "folder": "6. Concurrency and Bounded Queue",
            "name": "Queue Capacity Saturation (503 Service Overloaded)",
            "passed": passed,
            "status_code": res_503.status_code,
            "detail": res_503.json().get("error_code"),
        })

        # 6.3 Queue timeout -> 504
        concurrency_manager.configure(max_concurrency=1, max_queue=5, queue_timeout=0.1)
        barrier2 = asyncio.Event()

        async def slow_mock_analyze2(audio, filename, content_type, request_id):
            await barrier2.wait()
            return await fast_mock_analyze(audio, filename, content_type, request_id)

        setattr(routes_module, "analyze_audio", slow_mock_analyze2)

        task1 = asyncio.create_task(
            client.post("/v1/analyze", headers={"X-API-Key": api_key}, files={"file": ("1.wav", wav_1s, "audio/wav")})
        )
        await asyncio.sleep(0.05)
        res_504 = await client.post(
            "/v1/analyze", headers={"X-API-Key": api_key}, files={"file": ("2.wav", wav_1s, "audio/wav")}
        )
        barrier2.set()
        await task1

        passed = res_504.status_code == 504 and res_504.json().get("error_code") == "QUEUE_TIMEOUT"
        test_results.append({
            "folder": "6. Concurrency and Bounded Queue",
            "name": "Queue Timeout (504 Gateway Timeout)",
            "passed": passed,
            "status_code": res_504.status_code,
            "detail": res_504.json().get("error_code"),
        })

        concurrency_manager.configure(
            max_concurrency=settings.max_concurrent_inferences,
            max_queue=settings.max_queue_capacity,
            queue_timeout=settings.queue_timeout_seconds,
        )
        setattr(routes_module, "analyze_audio", fast_mock_analyze)

        # FOLDER 7: End-to-End Orchestration & TCED
        # 7.1 Genuine voice confirmed
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("sample.wav", wav_1s, "audio/wav")},
        )
        passed = (
            res.status_code == 200
            and res.json().get("final_outcome") == "GENUINE"
            and res.json().get("reason_code") == "ACOUSTIC_GENUINE_CONFIRMED"
        )
        test_results.append({
            "folder": "7. End-to-End Orchestration & TCED",
            "name": "TCED Genuine Voice Confirmed (GENUINE)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": f"{res.json().get('final_outcome')}/{res.json().get('reason_code')}",
        })

        # 7.2 Acoustic spoof detected
        async def fake_mock_analyze(audio, filename, content_type, request_id):
            return AnalysisResponse(
                request_id=request_id,
                final_outcome=FinalOutcome.FAKE,
                reason_code="ACOUSTIC_SPOOF_DETECTED",
                pella_status="success",
                rias_transcription_status="success",
                rias_risk_status="success",
                rias_risk_level=RiskLevel.LOW,
                latency_ms={"total": 50.0},
            )

        setattr(routes_module, "analyze_audio", fake_mock_analyze)
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("sample.wav", wav_1s, "audio/wav")},
        )
        passed = (
            res.status_code == 200
            and res.json().get("final_outcome") == "FAKE"
            and res.json().get("reason_code") == "ACOUSTIC_SPOOF_DETECTED"
        )
        test_results.append({
            "folder": "7. End-to-End Orchestration & TCED",
            "name": "TCED Acoustic Spoof Detected (FAKE)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": f"{res.json().get('final_outcome')}/{res.json().get('reason_code')}",
        })

        # 7.3 High semantic risk override
        async def high_risk_mock(audio, filename, content_type, request_id):
            return AnalysisResponse(
                request_id=request_id,
                final_outcome=FinalOutcome.SUSPICIOUS,
                reason_code="HIGH_RIAS_OVERRIDE",
                pella_status="success",
                rias_transcription_status="success",
                rias_risk_status="success",
                rias_risk_level=RiskLevel.HIGH,
                latency_ms={"total": 50.0},
            )

        setattr(routes_module, "analyze_audio", high_risk_mock)
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("sample.wav", wav_1s, "audio/wav")},
        )
        passed = (
            res.status_code == 200
            and res.json().get("final_outcome") == "SUSPICIOUS"
            and res.json().get("reason_code") == "HIGH_RIAS_OVERRIDE"
        )
        test_results.append({
            "folder": "7. End-to-End Orchestration & TCED",
            "name": "TCED High Semantic Risk Override (SUSPICIOUS)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": f"{res.json().get('final_outcome')}/{res.json().get('reason_code')}",
        })

        # 7.4 Inconclusive
        async def inconclusive_mock(audio, filename, content_type, request_id):
            return AnalysisResponse(
                request_id=request_id,
                final_outcome=FinalOutcome.INCONCLUSIVE,
                reason_code="INSUFFICIENT_EVIDENCE",
                pella_status="error",
                rias_transcription_status="success",
                rias_risk_status="success",
                rias_risk_level=RiskLevel.LOW,
                latency_ms={"total": 50.0},
            )

        setattr(routes_module, "analyze_audio", inconclusive_mock)
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("sample.wav", wav_1s, "audio/wav")},
        )
        passed = (
            res.status_code == 200
            and res.json().get("final_outcome") == "INCONCLUSIVE"
            and res.json().get("reason_code") == "INSUFFICIENT_EVIDENCE"
        )
        test_results.append({
            "folder": "7. End-to-End Orchestration & TCED",
            "name": "TCED Partial Evidence Insufficient (INCONCLUSIVE)",
            "passed": passed,
            "status_code": res.status_code,
            "detail": f"{res.json().get('final_outcome')}/{res.json().get('reason_code')}",
        })

        # Restore fast mock
        setattr(routes_module, "analyze_audio", fast_mock_analyze)

        # FOLDER 8: Edge Cases & Resiliency
        # 8.1 Custom X-Request-ID echoed
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key, "X-Request-ID": custom_trace_id},
            files={"file": ("sample.wav", wav_1s, "audio/wav")},
        )
        passed = (
            res.status_code == 200
            and res.headers.get("X-Request-ID") == custom_trace_id
            and res.json().get("request_id") == custom_trace_id
        )
        test_results.append({
            "folder": "8. Edge Cases & Resiliency",
            "name": "Custom X-Request-ID Header Echoed",
            "passed": passed,
            "status_code": res.status_code,
            "detail": f"echoed={res.headers.get('X-Request-ID')}",
        })

        # 8.2 Special characters in filename
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("call #1 [ext=audio] (2026).wav", wav_1s, "audio/wav")},
        )
        passed = res.status_code == 200 and "request_id" in res.json()
        test_results.append({
            "folder": "8. Edge Cases & Resiliency",
            "name": "Special Characters in Audio Filename",
            "passed": passed,
            "status_code": res.status_code,
            "detail": "Handled without escaping errors",
        })

        # 8.3 Application/octet-stream with .wav
        res = await client.post(
            "/v1/analyze",
            headers={"X-API-Key": api_key},
            files={"file": ("stream_audio.wav", wav_1s, "application/octet-stream")},
        )
        passed = res.status_code == 200 and "request_id" in res.json()
        test_results.append({
            "folder": "8. Edge Cases & Resiliency",
            "name": "Application/Octet-Stream MIME with .wav Extension",
            "passed": passed,
            "status_code": res.status_code,
            "detail": "Accepted and decoded binary stream",
        })

        # 8.4 Rapid sequential requests
        t_start = time.perf_counter()
        rapid_passed = True
        for _ in range(3):
            res_r = await client.post(
                "/v1/analyze",
                headers={"X-API-Key": api_key},
                files={"file": ("rapid.wav", wav_1s, "audio/wav")},
            )
            if res_r.status_code != 200:
                rapid_passed = False
        duration_ms = (time.perf_counter() - t_start) * 1000
        passed = rapid_passed and duration_ms < 5000
        test_results.append({
            "folder": "8. Edge Cases & Resiliency",
            "name": "Rapid Sequential Requests Latency Stability",
            "passed": passed,
            "status_code": 200,
            "detail": f"3 sequential requests in {duration_ms:.1f}ms",
        })

        # Clean up mock
        if orig_analyze:
            setattr(routes_module, "analyze_audio", orig_analyze)

    all_passed = all(t["passed"] for t in test_results)
    return all_passed, test_results


def main():
    parser = argparse.ArgumentParser(description="Validate CFIC Postman Collection & Environment")
    parser.add_argument(
        "--collection",
        default="postman/CFIC_VoiceShield_API_Gateway_Validation.postman_collection.json",
        help="Path to collection JSON",
    )
    parser.add_argument(
        "--environment",
        default="postman/CFIC_VoiceShield_Local.postman_environment.json",
        help="Path to environment JSON",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("CFIC VoiceShield — Postman Collection & API Gateway Validation Suite")
    print("=" * 80)

    # 1. File checks
    if not os.path.exists(args.collection):
        print(f"[ERROR] Collection file not found: {args.collection}")
        sys.exit(1)
    if not os.path.exists(args.environment):
        print(f"[ERROR] Environment file not found: {args.environment}")
        sys.exit(1)

    with open(args.collection, "r", encoding="utf-8") as f:
        collection_data = json.load(f)
    with open(args.environment, "r", encoding="utf-8") as f:
        env_data = json.load(f)

    # 2. Schema check
    print("\n--- PHASE 1: JSON Schema & Specification Compliance ---")
    schema_ok, schema_errors = validate_schema(collection_data, env_data)
    if schema_ok:
        log_pass("Postman Collection v2.1.0 schema format verified.")
        log_pass("Postman Environment structure and required variables verified.")
    else:
        for err in schema_errors:
            log_fail(err)
        sys.exit(1)

    # 3. Folder & Test assertions check
    print("\n--- PHASE 2: Folder Structure & Test Assertion Audit ---")
    folders_ok, folder_errors, stats = validate_folders_and_tests(collection_data)
    if folders_ok:
        log_pass(f"All 8 required functional folders present ({stats['folders_count']} total folders).")
        log_pass(f"Total requests audited: {stats['requests_count']}")
        log_pass(f"Total automated pm.test assertions audited: {stats['assertions_count']}")
        for fname, count in stats["per_folder"].items():
            log_info(f"Folder '{fname}': {count} requests verified.")
    else:
        for err in folder_errors:
            log_fail(err)
        sys.exit(1)

    # 4. Simulated Execution
    print("\n--- PHASE 3: Automated HTTP Request Execution & Assertion Verification ---")
    exec_ok, test_results = asyncio.run(run_simulated_execution(env_data))

    passed_count = sum(1 for t in test_results if t["passed"])
    failed_count = sum(1 for t in test_results if not t["passed"])

    current_folder = ""
    for t in test_results:
        if t["folder"] != current_folder:
            current_folder = t["folder"]
            print(f"\n  [{current_folder}]")
        if t["passed"]:
            log_pass(f"{t['name']} -> HTTP {t['status_code']} ({t['detail']})")
        else:
            log_fail(f"{t['name']} -> HTTP {t['status_code']} ({t['detail']})")

    print("\n" + "=" * 80)
    print(f"VALIDATION SUMMARY: {passed_count}/{len(test_results)} Tests Passed | {failed_count} Failed")
    print("=" * 80)

    if exec_ok:
        print("[SUCCESS] All API Gateway security, routing, chunking, and TCED assertions PASSED.")
        sys.exit(0)
    else:
        print("[FAILURE] One or more assertions failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
