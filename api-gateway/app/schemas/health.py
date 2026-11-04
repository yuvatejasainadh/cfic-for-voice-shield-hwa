"""Pydantic schemas for service health checks."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="ok")
    service: str = Field(default="cfic-api")
    version: str = Field(default="0.1.0")
    environment: str = Field(...)
    downstream_status: dict[str, str] = Field(default_factory=dict)
