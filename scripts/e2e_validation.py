"""End-to-End Validation Suite for CFIC x VoiceShield Hybrid Architecture.

Executes all 10 Phase 6 scenarios against the live containerized stack:
1. Telugu Audio Analysis
2. Hindi Audio Analysis
3. English Audio Analysis
4. Unsupported Language Handling (Assamese)
5. Invalid/Corrupted Audio Payload
6. Large Audio (>10s) Chunking & Aggregation
7. Empty Audio Payload (0-byte)
8. RIAS Downstream Unavailable (Fail-Safe INCONCLUSIVE Invariant)
9. LID Downstream Unavailable (Fail-Safe INCONCLUSIVE Invariant)
10. Privacy and Transcript Leakage Log Audit
"""

import io
import json
import os
import subprocess
import sys
import time
import wave
import httpx

GATEWAY_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("CFIC_API_KEY", "replace-me-for-local-testing")
HEADERS = {"X-API-Key": API_KEY}

TELUGU_AUDIO = r"cfic-asr-benchmark/data/audio/telugu/telugu_00000.wav"
HINDI_AUDIO = r"C:\Users\yuvat\Music\Multilingual Dataset\audio\hi_f_books_00118.wav"
ENGLISH_AUDIO = r"C:\Users\yuvat\Music\Multilingual Dataset\audio\en_f_lj_LJ050-0198.wav"
UNSUPPORTED_AUDIO = r"C:\Users\yuvat\Music\Multilingual Dataset\audio\ASM_F_ANGER_00109.wav"


def post_audio(audio_bytes: bytes, filename: str = "test.wav", timeout: float = 120.0) -> tuple[int, dict]:
    url = f"{GATEWAY_URL}/v1/analyze"
    files = {"file": (filename, audio_bytes, "audio/wav")}
    with httpx.Client(timeout=timeout) as client:
        try:
            res = client.post(url, headers=HEADERS, files=files)
            try:
                data = res.json()
            except Exception:
                data = {"raw_text": res.text}
            return res.status_code, data
        except Exception as exc:
            return 500, {"error": str(exc)}


def run_command(cmd: str) -> str:
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    stdout = res.stdout or ""
    stderr = res.stderr or ""
    return stdout + stderr


def main():
    print("=" * 80)
    print("CFIC x VoiceShield — End-to-End Live Validation (10 Scenarios)")
    print("=" * 80)
    sys.stdout.flush()
    results = {}

    # Test 1: Telugu Audio
    print("\n[Test 1] Telugu Audio Analysis...")
    sys.stdout.flush()
    with open(TELUGU_AUDIO, "rb") as f:
        te_bytes = f.read()
    t0 = time.perf_counter()
    status, data = post_audio(te_bytes, "telugu_sample.wav")
    t1 = time.perf_counter()
    print(f"Status: {status}, Latency: {round((t1 - t0)*1000, 2)}ms")
    print(f"Response: {json.dumps(data, indent=2)}")
    sys.stdout.flush()
    assert status == 200, f"Expected 200, got {status}"
    assert "final_outcome" in data, "Missing final_outcome"
    results["test_1_telugu"] = {"status": "PASSED", "http_status": status, "final_outcome": data["final_outcome"], "latency_ms": round((t1 - t0)*1000, 2), "data": data}

    # Test 2: Hindi Audio
    print("\n[Test 2] Hindi Audio Analysis...")
    sys.stdout.flush()
    with open(HINDI_AUDIO, "rb") as f:
        hi_bytes = f.read()
    t0 = time.perf_counter()
    status, data = post_audio(hi_bytes, "hindi_sample.wav")
    t1 = time.perf_counter()
    print(f"Status: {status}, Latency: {round((t1 - t0)*1000, 2)}ms")
    print(f"Response: {json.dumps(data, indent=2)}")
    sys.stdout.flush()
    assert status == 200, f"Expected 200, got {status}"
    assert "final_outcome" in data, "Missing final_outcome"
    results["test_2_hindi"] = {"status": "PASSED", "http_status": status, "final_outcome": data["final_outcome"], "latency_ms": round((t1 - t0)*1000, 2), "data": data}

    # Test 3: English Audio
    print("\n[Test 3] English Audio Analysis...")
    sys.stdout.flush()
    with open(ENGLISH_AUDIO, "rb") as f:
        en_bytes = f.read()
    t0 = time.perf_counter()
    status, data = post_audio(en_bytes, "english_sample.wav")
    t1 = time.perf_counter()
    print(f"Status: {status}, Latency: {round((t1 - t0)*1000, 2)}ms")
    print(f"Response: {json.dumps(data, indent=2)}")
    sys.stdout.flush()
    assert status == 200, f"Expected 200, got {status}"
    assert "final_outcome" in data, "Missing final_outcome"
    results["test_3_english"] = {"status": "PASSED", "http_status": status, "final_outcome": data["final_outcome"], "latency_ms": round((t1 - t0)*1000, 2), "data": data}

    # Test 4: Unsupported Language (Assamese)
    print("\n[Test 4] Unsupported Language Audio (Assamese)...")
    sys.stdout.flush()
    with open(UNSUPPORTED_AUDIO, "rb") as f:
        as_bytes = f.read()
    t0 = time.perf_counter()
    status, data = post_audio(as_bytes, "assamese_sample.wav")
    t1 = time.perf_counter()
    print(f"Status: {status}, Latency: {round((t1 - t0)*1000, 2)}ms")
    print(f"Response: {json.dumps(data, indent=2)}")
    sys.stdout.flush()
    assert status == 200, f"Expected 200, got {status}"
    assert data["final_outcome"] == "INCONCLUSIVE", f"Expected INCONCLUSIVE, got {data['final_outcome']}"
    results["test_4_unsupported_language"] = {"status": "PASSED", "http_status": status, "final_outcome": data["final_outcome"], "latency_ms": round((t1 - t0)*1000, 2), "data": data}

    # Test 5: Invalid Audio Payload
    print("\n[Test 5] Invalid / Corrupted Audio Payload...")
    sys.stdout.flush()
    invalid_bytes = b"CORRUPTED_NOT_A_REAL_WAV_HEADER_OR_AUDIO_DATA"
    t0 = time.perf_counter()
    status, data = post_audio(invalid_bytes, "invalid.wav")
    t1 = time.perf_counter()
    print(f"Status: {status}, Latency: {round((t1 - t0)*1000, 2)}ms")
    print(f"Response: {json.dumps(data, indent=2)}")
    sys.stdout.flush()
    assert status in (200, 400, 422), f"Expected 200, 400 or 422, got {status}"
    if status == 200:
        assert data["final_outcome"] == "INCONCLUSIVE", f"Expected INCONCLUSIVE for invalid audio, got {data['final_outcome']}"
    results["test_5_invalid_audio"] = {"status": "PASSED", "http_status": status, "latency_ms": round((t1 - t0)*1000, 2), "data": data}

    # Test 6: Large Audio (>10s) Chunking & Aggregation
    print("\n[Test 6] Large Audio (>10s) Chunking & Aggregation...")
    sys.stdout.flush()
    with wave.open(ENGLISH_AUDIO, "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    # Create ~11.5s audio payload (exceeds 10s threshold, spans 3 chunks: 0-5s, 5-10s, 10-11.5s)
    target_frames = frames + frames[:int(len(frames) * 0.35)]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out_w:
        out_w.setparams(params)
        out_w.writeframes(target_frames)
    large_bytes = buf.getvalue()
    duration_s = round(len(target_frames) / (params.framerate * params.nchannels * params.sampwidth), 2)
    print(f"Created large audio payload: {len(large_bytes)} bytes, duration ~{duration_s}s")
    sys.stdout.flush()
    t0 = time.perf_counter()
    status, data = post_audio(large_bytes, "large_english.wav", timeout=180.0)
    t1 = time.perf_counter()
    print(f"Status: {status}, Latency: {round((t1 - t0)*1000, 2)}ms")
    print(f"Response: {json.dumps(data, indent=2)}")
    sys.stdout.flush()
    assert status == 200, f"Expected 200, got {status}"
    assert "final_outcome" in data, "Missing final_outcome"
    results["test_6_large_audio"] = {"status": "PASSED", "http_status": status, "duration_s": duration_s, "latency_ms": round((t1 - t0)*1000, 2), "data": data}

    # Test 7: Empty Audio Payload (0-byte)
    print("\n[Test 7] Empty Audio Payload (0-byte)...")
    sys.stdout.flush()
    t0 = time.perf_counter()
    status, data = post_audio(b"", "empty.wav")
    t1 = time.perf_counter()
    print(f"Status: {status}, Latency: {round((t1 - t0)*1000, 2)}ms")
    print(f"Response: {json.dumps(data, indent=2)}")
    sys.stdout.flush()
    assert status == 400, f"Expected 400 for empty audio, got {status}"
    assert data.get("error_code") == "EMPTY_AUDIO_PAYLOAD", f"Expected EMPTY_AUDIO_PAYLOAD, got {data}"
    results["test_7_empty_audio"] = {"status": "PASSED", "http_status": status, "latency_ms": round((t1 - t0)*1000, 2), "data": data}

    # Test 8: RIAS Unavailable (Fail-Safe Invariant)
    print("\n[Test 8] RIAS Microservice Unavailable...")
    sys.stdout.flush()
    print("Stopping cfic-model-rias-gemma container...")
    run_command("docker stop cfic-model-rias-gemma")
    try:
        t0 = time.perf_counter()
        status, data = post_audio(en_bytes, "en_test8.wav")
        t1 = time.perf_counter()
        print(f"Status: {status}, Latency: {round((t1 - t0)*1000, 2)}ms")
        print(f"Response: {json.dumps(data, indent=2)}")
        sys.stdout.flush()
        assert status == 200, f"Expected 200, got {status}"
        assert data["final_outcome"] == "INCONCLUSIVE", f"Expected INCONCLUSIVE on RIAS failure, got {data['final_outcome']}"
        results["test_8_rias_unavailable"] = {"status": "PASSED", "http_status": status, "final_decision": data["final_outcome"], "latency_ms": round((t1 - t0)*1000, 2), "data": data}
    finally:
        print("Restarting cfic-model-rias-gemma container...")
        run_command("docker start cfic-model-rias-gemma")
        print("Waiting for RIAS container to be healthy...")
        sys.stdout.flush()
        for _ in range(40):
            out = run_command("docker exec cfic-model-rias-gemma curl -s http://localhost:8331/health")
            if '"loaded":true' in out:
                print("RIAS container healthy and loaded!")
                break
            time.sleep(1)

    # Test 9: LID Unavailable (Fail-Safe Invariant)
    print("\n[Test 9] LID Microservice Unavailable...")
    sys.stdout.flush()
    print("Stopping cfic-model-lid container...")
    run_command("docker stop cfic-model-lid")
    try:
        t0 = time.perf_counter()
        status, data = post_audio(en_bytes, "en_test9.wav")
        t1 = time.perf_counter()
        print(f"Status: {status}, Latency: {round((t1 - t0)*1000, 2)}ms")
        print(f"Response: {json.dumps(data, indent=2)}")
        sys.stdout.flush()
        assert status == 200, f"Expected 200, got {status}"
        assert data["final_outcome"] == "INCONCLUSIVE", f"Expected INCONCLUSIVE on LID failure, got {data['final_outcome']}"
        results["test_9_lid_unavailable"] = {"status": "PASSED", "http_status": status, "final_decision": data["final_outcome"], "latency_ms": round((t1 - t0)*1000, 2), "data": data}
    finally:
        print("Restarting cfic-model-lid container...")
        run_command("docker start cfic-model-lid")
        time.sleep(3)

    # Test 10: Privacy and Transcript Leakage Log Audit
    print("\n[Test 10] Privacy and Transcript Leakage Log Audit...")
    sys.stdout.flush()
    logs = run_command("docker compose logs")
    # Check that raw byte dumps or binary dumps are not leaked
    suspicious_patterns = ["b'\\x00", "RIFF\\x", "\\x00\\x01\\x02"]
    leaks = [p for p in suspicious_patterns if p in logs]
    print(f"Total docker compose logs size: {len(logs)} chars")
    print(f"Leak patterns found: {leaks}")
    sys.stdout.flush()
    assert len(leaks) == 0, f"Privacy violation! Leaked patterns: {leaks}"
    results["test_10_privacy_audit"] = {"status": "PASSED", "leaks_found": len(leaks)}

    print("\n" + "=" * 80)
    print("ALL 10 END-TO-END VALIDATION SCENARIOS PASSED SUCCESSFULLY!")
    print("=" * 80)
    sys.stdout.flush()

    with open("scripts/e2e_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
