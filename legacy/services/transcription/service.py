"""Transcription Service Scaffold.

Assigned Owner: Pallavi (Cloud Engineer).

NOTE: This file provides a functioning mock scaffold for service contract verification.
The mock must be replaced with the selected transcription model (Whisper Multilingual vs
AI4Bharat IndicConformer) following empirical benchmark results.
"""

import logging
import os
import time
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcription-service")

app = FastAPI(
    title="Speech-to-Text Transcription Service",
    version="0.1.0",
    description="Containerized transcription microservice supporting Indian multilingual audio",
)

MODEL_VERSION = os.getenv("TRANSCRIPTION_MODEL_VERSION", "whisper-indic-mock-v0.1")
MODEL_PATH = os.getenv("TRANSCRIPTION_MODEL_PATH", "models/transcription")
IS_MOCK = os.getenv("TRANSCRIPTION_USE_MOCK", "true").lower() == "true"


class TranscriptionResponse(BaseModel):
    status: str
    text: str | None
    language: str | None
    model_version: str
    latency_ms: float


class TranscriptionHealthResponse(BaseModel):
    status: str
    service: str
    model_version: str
    is_mock: bool
    artifact_found: bool


@app.get("/health", response_model=TranscriptionHealthResponse)
async def health():
    artifact_found = os.path.exists(MODEL_PATH)
    return TranscriptionHealthResponse(
        status="ok",
        service="transcription-service",
        model_version=MODEL_VERSION,
        is_mock=IS_MOCK,
        artifact_found=artifact_found,
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    start = time.perf_counter()
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file exceeds 25 MiB limit")

    if IS_MOCK:
        logger.warning(
            "[MOCK INFERENCE] Using mock transcription. Model selection pending Pallavi's benchmark."
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        # Mock transcription returns a realistic conversational sample for testing
        return TranscriptionResponse(
            status="success",
            text="Hello, this is an urgent security call from your bank regarding suspicious transactions.",
            language="en",
            model_version=MODEL_VERSION,
            latency_ms=elapsed_ms,
        )

    # Real model transcription pipeline (to be completed by Pallavi)
    # 1. Preprocess audio
    # 2. Forward pass through selected model (Whisper or IndicConformer)
    # 3. Decode transcript text and detect language
    raise HTTPException(
        status_code=503,
        detail="Real transcription model not loaded. Please provide model weights or set TRANSCRIPTION_USE_MOCK=true.",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8102)
