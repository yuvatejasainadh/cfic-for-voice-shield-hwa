"""Legacy RIAS assessor client shim delegating to risk_service."""

from app.services.risk_service import risk_service

assess_transcript = risk_service.assess_transcript

__all__ = ["assess_transcript"]
