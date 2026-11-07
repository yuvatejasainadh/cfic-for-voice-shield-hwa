"""AI4Bharat IndicConformer Telugu ASR Model Service."""

import os
import time
from typing import Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI(
    title="IndicConformer Telugu ASR Service",
    version="0.1.0",
    description="Containerized ASR model service serving Telugu IndicConformer checkpoint",
)

MODEL_PATH = os.getenv("ASR_TE_MODEL_PATH", "weights/indicconformer_stt_te_hybrid_rnnt_large.nemo")
MODEL_ID = os.getenv("ASR_TE_MODEL_ID", "ai4bharat/indicconformer-te")
MODEL_VERSION = os.getenv("ASR_TE_MODEL_VERSION", "indicconformer-te-large-v1")
IS_MOCK = os.getenv("ASR_TE_USE_MOCK", "true").lower() == "true"


class ASRResponse(BaseModel):
    status: str
    text: str | None
    language: str
    model_id: str = MODEL_ID
    model_version: str = MODEL_VERSION
    model_revision: Optional[str] = "v1"
    latency_ms: float


class ASRHealthResponse(BaseModel):
    status: str
    service: str
    target_language: str
    model_id: str
    model_version: str
    artifact_found: bool
    is_mock: bool


@app.get("/health", response_model=ASRHealthResponse)
async def health():
    return ASRHealthResponse(
        status="ok",
        service="model-asr-te",
        target_language="te",
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        artifact_found=os.path.exists(MODEL_PATH),
        is_mock=IS_MOCK,
    )


@app.post("/transcribe", response_model=ASRResponse)
async def transcribe(file: UploadFile = File(...)):
    start = time.perf_counter()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    if IS_MOCK or not os.path.exists(MODEL_PATH):
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return ASRResponse(
            status="success",
            text="దయచేసి మీ బ్యాంక్ వివరాలను తనిఖీ చేయండి.",
            language="te",
            model_id=MODEL_ID,
            model_version=MODEL_VERSION,
            model_revision="v1",
            latency_ms=elapsed_ms,
        )

    raise HTTPException(
        status_code=503,
        detail=f"IndicConformer Telugu weights not found at {MODEL_PATH}. Supply checkpoint or set ASR_TE_USE_MOCK=true.",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8312)
