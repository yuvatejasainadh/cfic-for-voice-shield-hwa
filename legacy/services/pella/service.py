"""PellaV2 Deepfake Detector Standalone Inference Service Scaffold.

Assigned Owner: Varshu (Containerization & Deployment) in collaboration with Varun (Model Preparation).

NOTE: This file provides a functioning mock scaffold for service contract verification.
The mock must be replaced with the real PellaV2 model loader once Varun verifies and hands off
the validated model artifact.
"""

import logging
import os
import time
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pella-service")

app = FastAPI(
    title="PellaV2 Deepfake Detection Service",
    version="0.1.0",
    description="Containerized inference microservice for PellaV2 audio deepfake detector",
)

MODEL_VERSION = os.getenv("PELLA_MODEL_VERSION", "pellav2-mock-placeholder")
MODEL_PATH = os.getenv("PELLA_MODEL_PATH", "models/pellav2")
IS_MOCK = os.getenv("PELLA_USE_MOCK", "true").lower() == "true"


class PellaPredictionResponse(BaseModel):
    status: str
    prediction: str | None
    score: float | None
    model_version: str
    latency_ms: float


class PellaHealthResponse(BaseModel):
    status: str
    service: str
    model_version: str
    is_mock: bool
    artifact_found: bool


@app.get("/health", response_model=PellaHealthResponse)
async def health():
    artifact_found = os.path.exists(MODEL_PATH)
    return PellaHealthResponse(
        status="ok",
        service="pella-service",
        model_version=MODEL_VERSION,
        is_mock=IS_MOCK,
        artifact_found=artifact_found,
    )


@app.post("/predict", response_model=PellaPredictionResponse)
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file exceeds 25 MiB limit")

    if IS_MOCK:
        logger.warning(
            "[MOCK INFERENCE] Using mock prediction. Replace with validated PellaV2 artifact."
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        # Mock returns a baseline placeholder
        return PellaPredictionResponse(
            status="success",
            prediction="GENUINE",
            score=0.15,
            model_version=MODEL_VERSION,
            latency_ms=elapsed_ms,
        )

    # Real model inference pipeline (to be completed by Varshu & Varun)
    # 1. Load audio bytes into memory / tensor
    # 2. Resample to 16 kHz mono (if required by artifact)
    # 3. Model forward pass
    # 4. Extract class score and label according to verified semantics
    raise HTTPException(
        status_code=503,
        detail="Real PellaV2 model not loaded. Please provide model weights or set PELLA_USE_MOCK=true.",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8101)
