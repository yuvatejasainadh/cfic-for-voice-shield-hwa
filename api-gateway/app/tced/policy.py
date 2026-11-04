"""Backward compatibility wrapper delegating to DecisionEngine."""

from app.orchestration.decision_engine import decide, decision_engine

__all__ = ["decide", "decision_engine"]
