"""Application-level job management and lifecycle tracking service."""

import time
import uuid
from typing import Any
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobService:
    """In-memory job tracking service (scaffolded for PostgreSQL / Redis transition)."""

    def __init__(self):
        self._jobs: dict[str, dict[str, Any]] = {}

    def create_job(self, request_id: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create a new analysis tracking job."""
        job_id = request_id or str(uuid.uuid4())
        job_record = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "created_at": time.time(),
            "updated_at": time.time(),
            "metadata": metadata or {},
            "result": None,
            "error": None,
        }
        self._jobs[job_id] = job_record
        return job_record

    def update_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        """Update job lifecycle status."""
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = status
            self._jobs[job_id]["updated_at"] = time.time()
            if error:
                self._jobs[job_id]["error"] = error

    def complete_job(self, job_id: str, result: dict[str, Any]) -> None:
        """Mark job completed with normalized result."""
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = JobStatus.COMPLETED
            self._jobs[job_id]["result"] = result
            self._jobs[job_id]["updated_at"] = time.time()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve job record by ID."""
        return self._jobs.get(job_id)


job_service = JobService()
