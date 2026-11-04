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
