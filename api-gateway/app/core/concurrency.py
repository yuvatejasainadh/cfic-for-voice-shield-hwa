"""Bounded concurrency and queue management for CFIC API Gateway inference requests."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from app.core.config import settings
from app.core.errors import QueueTimeoutException, ServiceOverloadedException

logger = logging.getLogger("concurrency_manager")


class ConcurrencyManager:
    """Manages active inference concurrency and bounded queue capacity."""

    def __init__(
        self,
        max_concurrency: int = 10,
        max_queue: int = 20,
        queue_timeout: float = 10.0,
    ):
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout

        self._semaphore = asyncio.Semaphore(0)
        self._lock = asyncio.Lock()
        self._active_count = 0
        self._waiting_count = 0

        # Diagnostics & Metrics
        self.total_processed = 0
        self.total_rejected = 0
        self.total_timeouts = 0

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def waiting_count(self) -> int:
        return self._waiting_count

    def configure(
        self,
        max_concurrency: int | None = None,
        max_queue: int | None = None,
        queue_timeout: float | None = None,
    ) -> None:
        """Dynamically configure limits and reset state."""
        if max_concurrency is not None:
            self.max_concurrency = max_concurrency
        if max_queue is not None:
            self.max_queue = max_queue
        if queue_timeout is not None:
            self.queue_timeout = queue_timeout
        self.reset()

    def reset(self) -> None:
        """Reset concurrency manager state (useful in test teardowns)."""
        self._semaphore = asyncio.Semaphore(0)
        self._active_count = 0
        self._waiting_count = 0
        self.total_processed = 0
        self.total_rejected = 0
        self.total_timeouts = 0

    @asynccontextmanager
    async def acquire_slot(self) -> AsyncGenerator[float, None]:
        """Context manager acquiring an execution slot under bounded concurrency.
        
        Yields queue_wait_ms (time spent waiting in queue).
        Raises ServiceOverloadedException (503) if queue capacity is exceeded.
        Raises QueueTimeoutException (504) if waiting exceeds timeout.
        """
        queue_start = time.perf_counter()

        async with self._lock:
            if self._active_count < self.max_concurrency:
                self._active_count += 1
                slot_acquired = True
            else:
                if self._waiting_count >= self.max_queue:
                    self.total_rejected += 1
                    logger.warning(
                        f"Inference queue saturated: {self._waiting_count} waiting >= {self.max_queue} max_queue, "
                        f"{self._active_count}/{self.max_concurrency} active. Rejecting request."
                    )
                    raise ServiceOverloadedException(
                        retry_after=5,
                        detail=f"Inference queue at capacity ({self._waiting_count}/{self.max_queue}). Please retry shortly.",
                    )
                self._waiting_count += 1
                slot_acquired = False

        if not slot_acquired:
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=self.queue_timeout)
            except asyncio.TimeoutError:
                async with self._lock:
                    self._waiting_count -= 1
                    self.total_timeouts += 1
                logger.warning(f"Inference queue timeout after {self.queue_timeout}s.")
                raise QueueTimeoutException(
                    f"Request timed out after {self.queue_timeout}s waiting for an inference execution slot."
                )

            async with self._lock:
                self._waiting_count -= 1
                self._active_count += 1

        queue_wait_ms = round((time.perf_counter() - queue_start) * 1000, 2)

        try:
            yield queue_wait_ms
        finally:
            async with self._lock:
                self._active_count -= 1
                self.total_processed += 1
                if self._waiting_count > 0:
                    self._semaphore.release()


# Default global concurrency manager configured from application settings
concurrency_manager = ConcurrencyManager(
    max_concurrency=settings.max_concurrent_inferences,
    max_queue=settings.max_queue_capacity,
    queue_timeout=settings.queue_timeout_seconds,
)
