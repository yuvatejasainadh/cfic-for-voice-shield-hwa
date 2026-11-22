"""Audio analysis ingestion and orchestration routes."""

import uuid
from fastapi import APIRouter, Depends, File, Header, Request, UploadFile
from app.core.concurrency import concurrency_manager
from app.core.config import settings
from app.core.errors import RateLimitExceededException, UnauthorizedException
from app.core.rate_limiter import InMemoryRateLimiter
from app.core.security import get_api_key_fingerprint, validate_api_key
from app.orchestration.analysis_orchestrator import analyze_audio
from app.schemas.analysis import AnalysisResponse
from app.services.audio_service import audio_service

router = APIRouter()

# Process-local rate limiter instance preserved for route dependency compatibility
rate_limiter = InMemoryRateLimiter(
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


def get_request_id(request: Request) -> str:
    """Retrieve existing X-Request-ID header or generate a new UUID."""
    req_id = request.headers.get("X-Request-ID")
    if not req_id:
        req_id = str(uuid.uuid4())
    return req_id


async def verify_api_key_and_rate_limit(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Validate API key authentication (constant-time) and enforce sliding-window rate limits."""
    # 1. API Key Authentication (constant-time comparison)
    if not validate_api_key(x_api_key):
        raise UnauthorizedException("Missing or invalid API key provided in X-API-Key header")

    # 2. Rate Limiting Key: Use client IP combined with SHA-256 API key fingerprint
    client_ip = request.client.host if request.client else "unknown"
    key_fingerprint = get_api_key_fingerprint(x_api_key)
    rate_key = f"{client_ip}:{key_fingerprint}"

    allowed, remaining, retry_after = rate_limiter.check(rate_key)
    if not allowed:
        raise RateLimitExceededException(retry_after=retry_after)


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Analyze audio sample for synthetic speech and conversational risk",
)
async def analyze(
    request: Request,
    file: UploadFile = File(..., description="Audio file payload (WAV/MP3/FLAC/OGG)"),
    _: None = Depends(verify_api_key_and_rate_limit),
) -> AnalysisResponse:
    """Accept an audio recording from VoiceShield, dispatch downstream analysis, and return decision outcome."""
    req_id = get_request_id(request)

    content = await file.read()
    filename = file.filename or "audio.wav"
    content_type = file.content_type or "audio/wav"

    # Audio Validation: MIME type, file extension, payload size, decodability, duration cap
    audio_service.validate_audio_payload(
        content=content,
        filename=filename,
        content_type=content_type,
    )

    import app.api.routes as routes_module
    analyze_fn = getattr(routes_module, "analyze_audio", analyze_audio)

    # Bounded Concurrency & Queue Management
    async with concurrency_manager.acquire_slot() as queue_wait_ms:
        response = await analyze_fn(
            audio=content,
            filename=filename,
            content_type=content_type,
            request_id=req_id,
        )

        if response.latency_ms is not None:
            response.latency_ms["queue_wait"] = queue_wait_ms

        return response
