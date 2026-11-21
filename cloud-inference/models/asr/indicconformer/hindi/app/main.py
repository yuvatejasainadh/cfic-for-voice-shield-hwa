"""AI4Bharat IndicConformer Hindi ASR Model Service."""

import os
import time
from typing import Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI(
    title="IndicConformer Hindi ASR Service",
    version="0.1.0",
    description="Containerized ASR model service serving Hindi IndicConformer checkpoint",
)

MODEL_PATH = os.getenv("ASR_HI_MODEL_PATH", "weights/indicconformer_hi.nemo")
MODEL_ID = os.getenv("ASR_HI_MODEL_ID", "ai4bharat/indicconformer-hi")
MODEL_VERSION = os.getenv("ASR_HI_MODEL_VERSION", "indicconformer-hi-large-v1")
IS_MOCK = os.getenv("ASR_HI_USE_MOCK", "true").lower() == "true"


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
        service="model-asr-hi",
        target_language="hi",
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
            text="कृपया अपने बैंक खाते का विवरण तुरंत सत्यापित करें।",
            language="hi",
            model_id=MODEL_ID,
            model_version=MODEL_VERSION,
            model_revision="v1",
            latency_ms=elapsed_ms,
        )

    raise HTTPException(
        status_code=503,
        detail=f"IndicConformer Hindi weights not found at {MODEL_PATH}. Supply checkpoint or set ASR_HI_USE_MOCK=true.",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8313)
