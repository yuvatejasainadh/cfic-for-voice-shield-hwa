"""Main FastAPI application entry point for CFIC API Gateway (HYBRID Architecture)."""

import uuid
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from app.api.routes import router
from app.core.config import settings
from app.core.errors import CFICException
from app.schemas.health import HealthResponse

app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description=(
        "CloudForge Intelligence Core (CFIC) API Gateway for VoiceShield (HYBRID Architecture). "
        "Provides client authentication, sliding-window rate limiting, audio payload validation, "
        "and coordinates multi-modal spoof detection and semantic risk analysis via Cloud Inference."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.middleware("http")
async def add_request_id_and_tracing(request: Request, call_next):
    """Ensure every request has an X-Request-ID header attached to both request state and response."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(CFICException)
async def cfic_exception_handler(request: Request, exc: CFICException):
    """Normalize CFIC exceptions into a consistent structured error schema."""
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    payload = {
        "error_code": exc.error_code,
        "detail": exc.detail,
        "request_id": request_id,
    }
    if exc.extra:
        payload["extra"] = exc.extra

    headers = dict(exc.headers)
    headers["X-Request-ID"] = request_id

    return JSONResponse(status_code=exc.status_code, content=payload, headers=headers)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Normalize standard HTTP exceptions."""
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    headers = dict(exc.headers or {})
    headers["X-Request-ID"] = request_id

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": f"HTTP_{exc.status_code}",
            "detail": exc.detail,
            "request_id": request_id,
        },
        headers=headers,
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def root_health() -> HealthResponse:
    """Root health check for load balancers and container orchestrators."""
    return HealthResponse(
        status="ok",
        service="cfic-api",
        version="0.2.0",
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


# Mount versioned API routes
app.include_router(router, prefix="/v1")
