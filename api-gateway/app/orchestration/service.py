"""Backward compatibility module for app.orchestration.service."""

from app.orchestration.analysis_orchestrator import (
    AnalysisOrchestrator,
    PipelineOrchestrator,
    analyze_audio,
    orchestrator,
)
from app.services.risk_service import assess_transcript

__all__ = [
    "AnalysisOrchestrator",
    "PipelineOrchestrator",
    "analyze_audio",
    "assess_transcript",
    "orchestrator",
]
