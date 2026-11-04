"""Authentication and client authorization service."""

from fastapi import Request
from app.core.config import settings
from app.core.errors import RateLimitExceededException, UnauthorizedException
from app.core.rate_limiter import InMemoryRateLimiter
from app.core.security import extract_api_key, validate_api_key

rate_limiter = InMemoryRateLimiter(
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


class AuthService:
    """Service governing API key authentication and rate limit enforcement."""

    def __init__(self, limiter: InMemoryRateLimiter | None = None):
        self.limiter = limiter or rate_limiter

    def authenticate_request(self, request: Request) -> str:
        """Authenticate request using X-API-Key header.
        
        Raises UnauthorizedException if missing or invalid.
        Returns the validated API key.
        """
        api_key = extract_api_key(request)
        if not validate_api_key(api_key):
            raise UnauthorizedException("Missing or invalid API key provided in X-API-Key header")
        return api_key  # type: ignore[return-value]

    def enforce_rate_limit(self, client_key: str) -> tuple[int, float]:
        """Enforce rate limits per client key.
        
        Raises RateLimitExceededException if quota is exceeded.
        Returns (remaining_requests, retry_after_seconds).
        """
        allowed, remaining, retry_after = self.limiter.check(client_key)
        if not allowed:
            raise RateLimitExceededException(retry_after=retry_after)
        return remaining, retry_after

    def verify_auth_and_rate_limit(self, request: Request) -> str:
        """Combined authentication and rate-limiting check."""
        api_key = self.authenticate_request(request)
        client_ip = request.client.host if request.client else "unknown_ip"
        client_identifier = f"{api_key}:{client_ip}"
        self.enforce_rate_limit(client_identifier)
        return api_key


auth_service = AuthService()
