"""Security, authentication and API key validation utilities."""

import hashlib
import secrets
from fastapi import Request
from app.core.config import settings
from app.core.errors import UnauthorizedException


def validate_api_key(provided_key: str | None) -> bool:
    """Validate incoming API key against configured settings.api_key using constant-time comparison.
    
    Returns True if valid, False otherwise.
    """
    if not provided_key or not isinstance(provided_key, str):
        return False
    # Guard against empty configured key in production
    expected_key = getattr(settings, "api_key", "")
    if not expected_key:
        return False
    return secrets.compare_digest(provided_key.strip(), expected_key.strip())


def get_api_key_fingerprint(api_key: str | None) -> str:
    """Generate a safe, irreversible SHA-256 fingerprint for rate-limit tracking without leaking key contents."""
    if not api_key:
        return "anonymous"
    return hashlib.sha256(api_key.strip().encode("utf-8", errors="ignore")).hexdigest()[:12]


def extract_api_key(request: Request) -> str | None:
    """Extract X-API-Key from request headers."""
    return request.headers.get("X-API-Key")


def enforce_api_key_auth(request: Request) -> str:
    """Enforce X-API-Key presence and validity, raising UnauthorizedException if invalid."""
    api_key = extract_api_key(request)
    if not validate_api_key(api_key):
        raise UnauthorizedException("Missing or invalid API key provided in X-API-Key header")
    return api_key  # type: ignore[return-value]
