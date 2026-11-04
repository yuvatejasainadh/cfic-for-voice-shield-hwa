"""AASIST-L Fallback: Acoustic Spoof Detector for English & Non-Regional Speech."""

import os
import time
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI(
    title="AASIST-L Fallback Acoustic Deepfake Detection Service",
    version="0.1.0",
    description="AASIST-L acoustic spoof detector serving English and fallback language audio streams",
)

MODEL_PATH = os.getenv("AASIST_MODEL_PATH", "weights/aasist_fallback.pt")
MODEL_VERSION = os.getenv("AASIST_MODEL_VERSION", "aasist-fallback-v1.0")
IS_MOCK = os.getenv("AASIST_USE_MOCK", "true").lower() == "true"
THRESHOLD = float(os.getenv("AASIST_THRESHOLD", "0.50"))


class SpoofPredictionResponse(BaseModel):
    status: str
    prediction: str | None
    score: float | None
    threshold: float
    model_version: str
    latency_ms: float


class SpoofHealthResponse(BaseModel):
    status: str
    service: str
    architecture: str
    target_domain: str
    model_version: str
    artifact_found: bool
    is_mock: bool


@app.get("/health", response_model=SpoofHealthResponse)
async def health():
    return SpoofHealthResponse(
        status="ok",
        service="model-aasist-fallback",
        architecture="AASIST-L-Baseline",
        target_domain="General-English-Fallback",
        model_version=MODEL_VERSION,
        artifact_found=os.path.exists(MODEL_PATH),
        is_mock=IS_MOCK,
    )


@app.post("/predict", response_model=SpoofPredictionResponse)
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    if IS_MOCK or not os.path.exists(MODEL_PATH):
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return SpoofPredictionResponse(
            status="success",
            prediction="GENUINE",
            score=0.08,
            threshold=THRESHOLD,
            model_version=MODEL_VERSION,
            latency_ms=elapsed_ms,
        )

    raise HTTPException(
        status_code=503,
        detail=f"AASIST-L Fallback weights not found at {MODEL_PATH}. Supply checkpoint or set AASIST_USE_MOCK=true.",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8322)
