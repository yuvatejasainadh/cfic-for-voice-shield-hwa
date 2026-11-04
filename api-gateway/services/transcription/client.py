"""Legacy TranscriptionClient shim delegating to api-gateway LegacyTranscriptionClient."""

from app.services.cfic_client import LegacyTranscriptionClient as TranscriptionClient

__all__ = ["TranscriptionClient"]
