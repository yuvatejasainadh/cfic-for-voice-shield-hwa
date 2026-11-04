"""API route aggregation for CFIC API Gateway."""

from fastapi import APIRouter
from app.api.routes.analyze import (
    analyze,
    analyze_audio,
    get_request_id,
    rate_limiter,
    router as analyze_router,
    verify_api_key_and_rate_limit,
)
from app.api.routes.health import health_check, router as health_router

router = APIRouter()
router.include_router(health_router)
router.include_router(analyze_router)

__all__ = [
    "analyze",
    "analyze_audio",
    "analyze_router",
    "get_request_id",
    "health_check",
    "health_router",
    "rate_limiter",
    "router",
    "verify_api_key_and_rate_limit",
]
