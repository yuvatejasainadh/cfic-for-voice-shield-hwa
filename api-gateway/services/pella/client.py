"""Legacy PellaClient shim delegating to api-gateway LegacyPellaClient."""

from app.services.cfic_client import LegacyPellaClient as PellaClient

__all__ = ["PellaClient"]
