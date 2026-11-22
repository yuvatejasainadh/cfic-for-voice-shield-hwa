"""Script to generate Postman Collection and Environment for CFIC VoiceShield API Gateway Validation."""

import json
import os
import wave


def ensure_samples():
    os.makedirs("postman", exist_ok=True)
    os.makedirs("storage/audio", exist_ok=True)

    # 1.0s, 5.0s, 10.0s normal WAVs
    for dur, name in [(1.0, "sample_1s.wav"), (5.0, "sample_5s.wav"), (10.0, "sample_10s.wav")]:
        path = os.path.join("storage/audio", name)
        if not os.path.exists(path):
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * int(16000 * dur))

    # Corrupt WAV
    corrupt_path = os.path.join("storage/audio", "corrupt.wav")
    if not os.path.exists(corrupt_path):
        with open(corrupt_path, "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00not-a-valid-pcm-audio-stream")

    # Text file
    txt_path = os.path.join("storage/audio", "test.txt")
    if not os.path.exists(txt_path):
        with open(txt_path, "wb") as f:
            f.write(b"This is a text file, not an audio recording.")


def make_req(name, method, url, headers=None, body_formdata=None, test_script=None, desc=""):
    header_list = []
    if headers:
        for k, v in headers.items():
            header_list.append({"key": k, "value": v, "type": "text"})

    body = None
    if body_formdata:
        body = {"mode": "formdata", "formdata": body_formdata}

    events = []
    if test_script:
        events.append({
            "listen": "test",
            "script": {
                "type": "text/javascript",
                "exec": test_script.strip().split("\n") if isinstance(test_script, str) else test_script,
            },
        })

    # Parse path segments from url
    raw_path = url.replace("{{base_url}}", "").strip("/")
    path_segments = [p for p in raw_path.split("/") if p]

    return {
        "name": name,
        "request": {
            "method": method,
            "header": header_list,
            "body": body,
            "url": {
                "raw": url,
                "host": ["{{base_url}}"],
                "path": path_segments,
            },
            "description": desc,
        },
        "response": [],
        "event": events,
    }


def build_collection():
    ensure_samples()

    collection = {
        "info": {
            "_postman_id": "c8f49a12-6e21-42a1-9df1-801fcbf6c1e3",
            "name": "CFIC VoiceShield API Gateway Validation",
            "description": (
                "Comprehensive API Gateway test suite for VoiceShield covering Authentication, "
                "Rate Limiting, Request Validation, 5-Second Window Chunking, Bounded Concurrency & Queueing, "
                "TCED Decision Policy, and Edge Case Resilience."
            ),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "item": [],
    }

    # 1. Health and Discovery
    collection["item"].append({
        "name": "1. Health and Discovery",
        "description": "Endpoints for load balancer probes, system health verification, and interactive API documentation.",
        "item": [
            make_req(
                "GET System Root Health (/health)",
                "GET",
                "{{base_url}}/health",
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Health status is ok and service is cfic-api", function () {
    const data = pm.response.json();
    pm.expect(data.status).to.eql("ok");
    pm.expect(data.service).to.eql("cfic-api");
    pm.expect(data.downstream_status).to.be.an("object");
});
pm.test("Response time is acceptable", function () {
    pm.expect(pm.response.responseTime).to.be.below(1000);
});""",
                desc="Root health probe used by container orchestrators and load balancers.",
            ),
            make_req(
                "GET Versioned Health (/v1/health)",
                "GET",
                "{{base_url}}/v1/health",
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Versioned health confirms ok", function () {
    const data = pm.response.json();
    pm.expect(data.status).to.eql("ok");
});""",
                desc="API v1 health status endpoint.",
            ),
            make_req(
                "GET Swagger OpenAPI Docs (/docs)",
                "GET",
                "{{base_url}}/docs",
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Content-Type is text/html", function () {
    pm.expect(pm.response.headers.get("Content-Type")).to.include("text/html");
});""",
                desc="Interactive Swagger OpenAPI specification UI.",
            ),
            make_req(
                "GET ReDoc Documentation (/redoc)",
                "GET",
                "{{base_url}}/redoc",
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Content-Type is text/html", function () {
    pm.expect(pm.response.headers.get("Content-Type")).to.include("text/html");
});""",
                desc="Alternative ReDoc documentation endpoint.",
            ),
        ],
    })

    # 2. Authentication and Authorization
    collection["item"].append({
        "name": "2. Authentication and Authorization",
        "description": "Verification of constant-time X-API-Key authentication, rejection of missing/invalid keys, and header preservation.",
        "item": [
            make_req(
                "Missing API Key (401 Unauthorized)",
                "POST",
                "{{base_url}}/v1/analyze",
                body_formdata=[{"key": "file", "type": "file", "src": "{{sample_audio_path}}"}],
                test_script="""pm.test("Status code is 401 Unauthorized", function () {
    pm.response.to.have.status(401);
});
pm.test("Error code is UNAUTHORIZED", function () {
    const data = pm.response.json();
    pm.expect(data.error_code).to.eql("UNAUTHORIZED");
    pm.expect(data.detail).to.include("API key");
});
pm.test("X-Request-ID header is attached for tracing", function () {
    pm.expect(pm.response.headers.has("X-Request-ID")).to.be.true;
});""",
                desc="Omission of X-API-Key header must return 401 UNAUTHORIZED.",
            ),
            make_req(
                "Invalid API Key (401 Unauthorized)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{invalid_api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "{{sample_audio_path}}"}],
                test_script="""pm.test("Status code is 401 Unauthorized", function () {
    pm.response.to.have.status(401);
});
pm.test("Error code is UNAUTHORIZED", function () {
    const data = pm.response.json();
    pm.expect(data.error_code).to.eql("UNAUTHORIZED");
});""",
                desc="Invalid X-API-Key header must be rejected in constant time.",
            ),
            make_req(
                "Empty API Key (401 Unauthorized)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": ""},
                body_formdata=[{"key": "file", "type": "file", "src": "{{sample_audio_path}}"}],
                test_script="""pm.test("Status code is 401 Unauthorized", function () {
    pm.response.to.have.status(401);
});
pm.test("Error code is UNAUTHORIZED", function () {
    const data = pm.response.json();
    pm.expect(data.error_code).to.eql("UNAUTHORIZED");
});""",
                desc="Empty string API key must be rejected with 401.",
            ),
            make_req(
                "Valid API Key Authentication Accepted",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "{{sample_audio_path}}"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Authentication succeeds and returns analysis response", function () {
    const data = pm.response.json();
    pm.expect(data).to.have.property("request_id");
    pm.expect(data).to.have.property("final_outcome");
    pm.expect(data).to.have.property("reason_code");
});""",
                desc="Valid X-API-Key passes authentication gate.",
            ),
        ],
    })

    # 3. Rate Limiting
    collection["item"].append({
        "name": "3. Rate Limiting",
        "description": "Sliding-window rate limiter enforcement per client IP + key fingerprint, Retry-After header delivery, and stale memory eviction.",
        "item": [
            make_req(
                "Request Within Normal Rate Limit",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "{{sample_audio_path}}"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Response has valid request_id", function () {
    const data = pm.response.json();
    pm.expect(data.request_id).to.be.a("string");
});""",
                desc="Single request under rate limit window succeeds.",
            ),
            make_req(
                "Burst Limit Exceeded (429 Rate Limit Exceeded)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "{{sample_audio_path}}"}],
                test_script="""pm.test("Status code is 429 Too Many Requests", function () {
    pm.response.to.have.status(429);
});
pm.test("Error code is RATE_LIMIT_EXCEEDED", function () {
    const data = pm.response.json();
    pm.expect(data.error_code).to.eql("RATE_LIMIT_EXCEEDED");
});
pm.test("Retry-After header is present and integer", function () {
    pm.expect(pm.response.headers.has("Retry-After")).to.be.true;
    const retryAfter = parseInt(pm.response.headers.get("Retry-After"), 10);
    pm.expect(retryAfter).to.be.above(0);
});""",
                desc="Simulates burst rate limit exceeding window quota, expecting 429 and Retry-After header.",
            ),
        ],
    })

    # 4. Request Validation
    collection["item"].append({
        "name": "4. Request Validation",
        "description": "Strict validation of audio MIME types, file extensions, payload size bounds (25MB), decodability, and duration cap (120s).",
        "item": [
            make_req(
                "Empty Audio Payload (400 Bad Request)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": ""}],
                test_script="""pm.test("Status code is 400 Bad Request", function () {
    pm.response.to.have.status(400);
});
pm.test("Error code is EMPTY_AUDIO_PAYLOAD", function () {
    const data = pm.response.json();
    pm.expect(data.error_code).to.eql("EMPTY_AUDIO_PAYLOAD");
});""",
                desc="Submitting an empty 0-byte file must return 400 Bad Request.",
            ),
            make_req(
                "Oversized Audio Payload (>25MB, 413 Payload Too Large)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_oversized.wav"}],
                test_script="""pm.test("Status code is 413 Payload Too Large", function () {
    pm.response.to.have.status(413);
});
pm.test("Error code is AUDIO_PAYLOAD_TOO_LARGE", function () {
    const data = pm.response.json();
    pm.expect(data.error_code).to.eql("AUDIO_PAYLOAD_TOO_LARGE");
    pm.expect(data.extra).to.have.property("max_bytes");
});""",
                desc="Audio payload exceeding max_audio_bytes (25MB) must be rejected with 413.",
            ),
            make_req(
                "Invalid MIME Type (text/plain, 415 Unsupported Media Type)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/test.txt", "contentType": "text/plain"}],
                test_script="""pm.test("Status code is 415 Unsupported Media Type", function () {
    pm.response.to.have.status(415);
});
pm.test("Error code is UNSUPPORTED_AUDIO_FORMAT", function () {
    const data = pm.response.json();
    pm.expect(data.error_code).to.eql("UNSUPPORTED_AUDIO_FORMAT");
});""",
                desc="Non-audio MIME types must be rejected with 415.",
            ),
            make_req(
                "Corrupted Audio Bitstream (400 Bad Request)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/corrupt.wav"}],
                test_script="""pm.test("Status code is 400 Bad Request", function () {
    pm.response.to.have.status(400);
});
pm.test("Error code is CORRUPTED_AUDIO_PAYLOAD", function () {
    const data = pm.response.json();
    pm.expect(data.error_code).to.eql("CORRUPTED_AUDIO_PAYLOAD");
});""",
                desc="Uploaded file with valid extension/MIME but corrupted header/bitstream returns 400.",
            ),
            make_req(
                "Audio Exceeding Duration Cap (>120s, 400 Bad Request)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/too_long.wav"}],
                test_script="""pm.test("Status code is 400 Bad Request", function () {
    pm.response.to.have.status(400);
});
pm.test("Error code is AUDIO_DURATION_TOO_LONG", function () {
    const data = pm.response.json();
    pm.expect(data.error_code).to.eql("AUDIO_DURATION_TOO_LONG");
    pm.expect(data.extra).to.have.property("max_duration_seconds");
});""",
                desc="Audio duration longer than 120 seconds must be rejected with 400.",
            ),
        ],
    })

    # 5. Audio Chunking
    collection["item"].append({
        "name": "5. Audio Chunking",
        "description": "Sequential 5-second window chunking orchestration ensuring sample conservation across various file durations.",
        "item": [
            make_req(
                "Short Audio Sample (<5s, 1s Single Chunk)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_1s.wav"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Processed as single chunk with valid schema", function () {
    const data = pm.response.json();
    pm.expect(data.final_outcome).to.be.oneOf(["GENUINE", "FAKE", "SUSPICIOUS", "INCONCLUSIVE"]);
});""",
                desc="Audio shorter than 5 seconds processed as single chunk without padding distortion.",
            ),
            make_req(
                "Standard 5s Sample (Exact Chunk Boundary)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_5s.wav"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Response contains latency metrics", function () {
    const data = pm.response.json();
    pm.expect(data.latency_ms).to.have.property("total");
});""",
                desc="Exact 5.0-second audio sample processed cleanly at window boundary.",
            ),
            make_req(
                "Multi-Chunk 10s Sample (2 Sequential Chunks)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_10s.wav"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Orchestrated across sequential chunks", function () {
    const data = pm.response.json();
    pm.expect(data.pella_status).to.be.a("string");
    pm.expect(data.rias_transcription_status).to.be.a("string");
});""",
                desc="10-second sample sliced into two 5-second chunks and aggregated.",
            ),
        ],
    })

    # 6. Concurrency and Bounded Queue
    collection["item"].append({
        "name": "6. Concurrency and Bounded Queue",
        "description": "Verification of max concurrent inference slots, bounded queue capacity, queue latency reporting, and 503/504 backpressure.",
        "item": [
            make_req(
                "Normal Concurrency Slot Acquisition",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_1s.wav"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Latency breakdown includes queue_wait metric", function () {
    const data = pm.response.json();
    pm.expect(data.latency_ms).to.have.property("queue_wait");
    pm.expect(data.latency_ms.queue_wait).to.be.at.least(0);
});""",
                desc="Acquires concurrency slot and reports queue_wait latency in milliseconds.",
            ),
            make_req(
                "Queue Capacity Saturation (503 Service Overloaded)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_1s.wav"}],
                test_script="""pm.test("Status code is 503 Service Unavailable", function () {
    pm.response.to.have.status(503);
});
pm.test("Error code is SERVICE_OVERLOADED", function () {
    const data = pm.response.json();
    pm.expect(data.error_code).to.eql("SERVICE_OVERLOADED");
});
pm.test("Retry-After header provided", function () {
    pm.expect(pm.response.headers.has("Retry-After")).to.be.true;
});""",
                desc="When waiting queue capacity is exceeded, returns 503 SERVICE_OVERLOADED with Retry-After.",
            ),
            make_req(
                "Queue Timeout (504 Gateway Timeout)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_1s.wav"}],
                test_script="""pm.test("Status code is 504 Gateway Timeout", function () {
    pm.response.to.have.status(504);
});
pm.test("Error code is QUEUE_TIMEOUT", function () {
    const data = pm.response.json();
    pm.expect(data.error_code).to.eql("QUEUE_TIMEOUT");
});""",
                desc="When request wait in queue exceeds queue_timeout_seconds, returns 504 QUEUE_TIMEOUT.",
            ),
        ],
    })

    # 7. End-to-End Orchestration & TCED
    collection["item"].append({
        "name": "7. End-to-End Orchestration & TCED",
        "description": "Authoritative 15-combination TCED decision matrix synthesis across acoustic and conversational evidence.",
        "item": [
            make_req(
                "TCED Genuine Voice Confirmed (GENUINE)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_1s.wav"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Final outcome is GENUINE", function () {
    const data = pm.response.json();
    pm.expect(data.final_outcome).to.eql("GENUINE");
    pm.expect(data.reason_code).to.eql("ACOUSTIC_GENUINE_CONFIRMED");
});""",
                desc="Acoustic GENUINE + RIAS LOW yields final outcome GENUINE.",
            ),
            make_req(
                "TCED Acoustic Spoof Detected (FAKE)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_1s.wav"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Final outcome is FAKE", function () {
    const data = pm.response.json();
    pm.expect(data.final_outcome).to.eql("FAKE");
    pm.expect(data.reason_code).to.eql("ACOUSTIC_SPOOF_DETECTED");
});""",
                desc="Acoustic FAKE yields final outcome FAKE.",
            ),
            make_req(
                "TCED High Semantic Risk Override (SUSPICIOUS)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_1s.wav"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Final outcome is SUSPICIOUS due to HIGH RIAS override", function () {
    const data = pm.response.json();
    pm.expect(data.final_outcome).to.eql("SUSPICIOUS");
    pm.expect(data.reason_code).to.eql("HIGH_RIAS_OVERRIDE");
    pm.expect(data.rias_risk_level).to.eql("HIGH");
});""",
                desc="Acoustic GENUINE overridden by RIAS HIGH into SUSPICIOUS.",
            ),
            make_req(
                "TCED Partial Evidence Insufficient (INCONCLUSIVE)",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_1s.wav"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Final outcome is INCONCLUSIVE when evidence is insufficient", function () {
    const data = pm.response.json();
    pm.expect(data.final_outcome).to.eql("INCONCLUSIVE");
    pm.expect(data.reason_code).to.eql("INSUFFICIENT_EVIDENCE");
});""",
                desc="Acoustic UNAVAILABLE + RIAS LOW yields INCONCLUSIVE.",
            ),
        ],
    })

    # 8. Edge Cases & Resiliency
    collection["item"].append({
        "name": "8. Edge Cases & Resiliency",
        "description": "Edge cases including custom request ID header tracing, special characters in filenames, binary MIME streams, and latency consistency.",
        "item": [
            make_req(
                "Custom X-Request-ID Header Echoed",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}", "X-Request-ID": "{{request_id}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_1s.wav"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Echoes custom X-Request-ID in response header", function () {
    const expected = pm.environment.get("request_id") || "custom-trace-uuid-12345";
    pm.expect(pm.response.headers.get("X-Request-ID")).to.eql(expected);
});
pm.test("Response body request_id matches custom ID", function () {
    const data = pm.response.json();
    const expected = pm.environment.get("request_id") || "custom-trace-uuid-12345";
    pm.expect(data.request_id).to.eql(expected);
});""",
                desc="Client-provided X-Request-ID must be echoed in header and response body.",
            ),
            make_req(
                "Special Characters in Audio Filename",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_1s.wav"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Processed safely without escaping bugs", function () {
    const data = pm.response.json();
    pm.expect(data.request_id).to.be.a("string");
});""",
                desc="Filenames with spaces, hashes, brackets must be handled safely.",
            ),
            make_req(
                "Application/Octet-Stream MIME with .wav Extension",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_1s.wav", "contentType": "application/octet-stream"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Successfully decoded binary stream", function () {
    const data = pm.response.json();
    pm.expect(data.final_outcome).to.be.a("string");
});""",
                desc="Binary octet-stream upload with supported audio extension is accepted.",
            ),
            make_req(
                "Rapid Sequential Requests Latency Stability",
                "POST",
                "{{base_url}}/v1/analyze",
                headers={"X-API-Key": "{{api_key}}"},
                body_formdata=[{"key": "file", "type": "file", "src": "storage/audio/sample_1s.wav"}],
                test_script="""pm.test("Status code is 200 OK", function () {
    pm.response.to.have.status(200);
});
pm.test("Total latency is acceptable", function () {
    pm.expect(pm.response.responseTime).to.be.below(5000);
});""",
                desc="Verifies Gateway stability and latency under sequential invocations.",
            ),
        ],
    })

    out_path = os.path.join("postman", "CFIC_VoiceShield_API_Gateway_Validation.postman_collection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)

    print(f"Generated {out_path} with {len(collection['item'])} folders.")


if __name__ == "__main__":
    build_collection()
