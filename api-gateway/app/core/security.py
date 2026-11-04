"""Security, authentication and API key validation utilities."""

from fastapi import Request
from app.core.config import settings
from app.core.errors import UnauthorizedException


def validate_api_key(provided_key: str | None) -> bool:
    """Validate incoming API key against configured settings.api_key.
    
    Returns True if valid, False otherwise.
    """
    if not provided_key:
        return False
    return provided_key == settings.api_key


def extract_api_key(request: Request) -> str | None:
    """Extract X-API-Key from request headers."""
    return request.headers.get("X-API-Key")


def enforce_api_key_auth(request: Request) -> str:
    """Enforce X-API-Key presence and validity, raising UnauthorizedException if invalid."""
    api_key = extract_api_key(request)
    if not validate_api_key(api_key):
        raise UnauthorizedException("Missing or invalid API key provided in X-API-Key header")
    return api_key  # type: ignore[return-value]
