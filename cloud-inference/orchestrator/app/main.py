"""Cloud Inference Orchestrator — Central model-serving coordinator."""

from fastapi import FastAPI
from app.api.routes import router
from app.core.config import settings

app = FastAPI(
    title=settings.service_name,
    version="0.1.0",
    description=(
        "Cloud Inference Orchestrator coordinating Language Identification (LID), "
        "Language-specific IndicConformer ASR, and Acoustic Spoof Detection (VMC-South / AASIST-L)."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
