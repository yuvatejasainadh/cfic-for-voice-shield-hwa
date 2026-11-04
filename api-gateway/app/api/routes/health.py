"""Health and system status routes."""

from fastapi import APIRouter
from app.core.config import settings
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health status and downstream dependency configuration."""
    return HealthResponse(
        status="ok",
        service="cfic-api",
        version="0.1.0",
        environment=settings.environment,
        downstream_status={
            "cloud_inference_configured": str(bool(settings.cloud_inference_url)),
            "pella_url_configured": str(bool(settings.pella_base_url)),
            "transcription_url_configured": str(bool(settings.transcription_base_url)),
            "askdrip_llm_configured": str(
                bool(settings.askdrip_llm_base_url and settings.askdrip_llm_api_key)
            ),
        },
    )
