# CFIC API Gateway — API Contracts Specification

**Base URL:** `http://localhost:8000`  
**Authentication:** `X-API-Key` HTTP Header  
**Tracing:** `X-Request-ID` HTTP Header (propagated across all downstream calls)

---

## 1. `GET /health` and `GET /v1/health`
System status and downstream dependency verification.

- **Response (200 OK):**
```json
{
  "status": "ok",
  "service": "cfic-api",
  "version": "0.2.0",
  "environment": "development",
  "downstream_status": {
    "cloud_inference_configured": "True",
    "pella_url_configured": "True",
    "transcription_url_configured": "True",
    "askdrip_llm_configured": "False"
  }
}
```

---

## 2. `POST /v1/analyze`
Main ingestion endpoint for VoiceShield client call audio.

- **Headers:**
  - `X-API-Key` *(required)*: Authentication key matching `CFIC_API_KEY`.
  - `X-Request-ID` *(optional)*: UUID tracing identifier.
- **Request:**
  - Content-Type: `multipart/form-data`
  - Body: `file: UploadFile` (WAV, MP3, OGG, FLAC, binary audio).
- **Constraints:**
  - Max audio file size: 25 MiB (`CFIC_MAX_AUDIO_BYTES`).
  - Rate Limit: 30 requests / 60 seconds per client key.
- **Success Response (200 OK):**
```json
{
  "request_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "final_outcome": "SUSPICIOUS",
  "reason_code": "HIGH_RIAS_OVERRIDE",
  "pella_status": "success",
  "rias_transcription_status": "success",
  "rias_risk_status": "success",
  "rias_risk_level": "HIGH",
  "model_version": "vmc-south-v1.0",
  "latency_ms": {
    "pella": 38.5,
    "transcription": 115.0,
    "rias_llm": 210.0,
    "total": 365.2
  }
}
```

- **Error Codes:**
  - `401 Unauthorized`: Missing or invalid API key.
  - `400 Bad Request`: Empty audio payload.
  - `413 Payload Too Large`: Upload exceeds size limit.
  - `429 Too Many Requests`: Rate limit exceeded (`Retry-After` header present).
