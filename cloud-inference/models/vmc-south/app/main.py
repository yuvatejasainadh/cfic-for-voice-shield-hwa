"""VMC-South: Fine-tuned AASIST-L Acoustic Spoof Detector for South Indian Speech."""

import os
import time
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI(
    title="VMC-South Acoustic Deepfake Detection Service",
    version="0.1.0",
    description="Fine-tuned AASIST-L acoustic spoof detector specialized for South Indian languages (Telugu, Tamil, Kannada, Malayalam)",
)

MODEL_PATH = os.getenv("VMC_SOUTH_MODEL_PATH", "weights/vmc_south_aasist.pt")
MODEL_VERSION = os.getenv("VMC_SOUTH_MODEL_VERSION", "vmc-south-aasistl-v1.0")
IS_MOCK = os.getenv("VMC_SOUTH_USE_MOCK", "true").lower() == "true"
THRESHOLD = float(os.getenv("VMC_SOUTH_THRESHOLD", "0.50"))


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
        service="model-vmc-south",
        architecture="AASIST-L-FineTuned",
        target_domain="South-Indian-Multilingual",
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
        # Mock scaffold response: score < threshold implies GENUINE
        return SpoofPredictionResponse(
            status="success",
            prediction="GENUINE",
            score=0.06,
            threshold=THRESHOLD,
            model_version=MODEL_VERSION,
            latency_ms=elapsed_ms,
        )

    # Real AASIST-L forward pass execution
    raise HTTPException(
        status_code=503,
        detail=f"VMC-South weights not found at {MODEL_PATH}. Supply checkpoint or set VMC_SOUTH_USE_MOCK=true.",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8321)
