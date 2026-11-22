"""Structured errors and exception definitions for CFIC API Gateway."""

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Normalized structured error response."""

    error_code: str = Field(..., description="Machine-readable error identifier")
    detail: str = Field(..., description="Human-readable explanation of the error")
    request_id: str = Field(..., description="Unique request tracing ID")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )
    extra: dict[str, Any] | None = Field(
        default=None, description="Optional safe contextual details"
    )


class CFICException(Exception):
    """Base exception for CFIC errors."""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        detail: str,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ):
        super().__init__(detail)
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail
        self.headers = headers or {}
        self.extra = extra


class UnauthorizedException(CFICException):
    def __init__(self, detail: str = "Invalid or missing API key"):
        super().__init__(
            status_code=401,
            error_code="UNAUTHORIZED",
            detail=detail,
            headers={"WWW-Authenticate": "ApiKey"},
        )


class EmptyAudioPayloadException(CFICException):
    def __init__(self, detail: str = "Empty audio payload provided"):
        super().__init__(
            status_code=400,
            error_code="EMPTY_AUDIO_PAYLOAD",
            detail=detail,
        )


class AudioPayloadTooLargeException(CFICException):
    def __init__(self, max_bytes: int, actual_bytes: int):
        detail = (
            f"Audio payload size ({actual_bytes} bytes) exceeds limit of {max_bytes} bytes"
        )
        super().__init__(
            status_code=413,
            error_code="AUDIO_PAYLOAD_TOO_LARGE",
            detail=detail,
            extra={"max_bytes": max_bytes, "actual_bytes": actual_bytes},
        )


class RateLimitExceededException(CFICException):
    def __init__(self, retry_after: float):
        detail = f"Rate limit exceeded. Try again in {retry_after} seconds."
        super().__init__(
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            detail=detail,
            headers={"Retry-After": str(int(retry_after) + 1)},
            extra={"retry_after_seconds": retry_after},
        )


class CorruptedAudioPayloadException(CFICException):
    def __init__(self, detail: str = "Audio payload is corrupted or contains unparseable bitstream"):
        super().__init__(
            status_code=400,
            error_code="CORRUPTED_AUDIO_PAYLOAD",
            detail=detail,
        )


class UnsupportedAudioFormatException(CFICException):
    def __init__(self, detail: str = "Unsupported audio encoding or MIME type"):
        super().__init__(
            status_code=415,
            error_code="UNSUPPORTED_AUDIO_FORMAT",
            detail=detail,
        )


class AudioDurationTooLongException(CFICException):
    def __init__(self, max_duration_s: float, actual_duration_s: float):
        detail = f"Audio duration ({actual_duration_s:.1f}s) exceeds maximum allowed duration of {max_duration_s:.1f}s"
        super().__init__(
            status_code=400,
            error_code="AUDIO_DURATION_TOO_LONG",
            detail=detail,
            extra={"max_duration_seconds": max_duration_s, "actual_duration_seconds": actual_duration_s},
        )


class ServiceOverloadedException(CFICException):
    def __init__(self, retry_after: int = 5, detail: str = "Inference service is currently overloaded. Please retry shortly."):
        super().__init__(
            status_code=503,
            error_code="SERVICE_OVERLOADED",
            detail=detail,
            headers={"Retry-After": str(retry_after)},
            extra={"retry_after_seconds": retry_after},
        )


class QueueTimeoutException(CFICException):
    def __init__(self, detail: str = "Request timed out while waiting in queue for inference worker"):
        super().__init__(
            status_code=504,
            error_code="QUEUE_TIMEOUT",
            detail=detail,
        )
