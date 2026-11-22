"""In-memory rate limiter for CFIC API Gateway.

Note on multi-instance scaling:
This implementation is process-local and suitable for single-instance deployments,
local development, and containerized demo environments.
For multi-instance, clustered production deployments behind a load balancer,
infrastructure/redis contains configuration for Redis-backed distributed rate limiting.
"""

import time
from collections import defaultdict
from threading import Lock


class InMemoryRateLimiter:
    """Sliding-window in-memory rate limiter with bounded memory tracking."""

    def __init__(
        self,
        max_requests: int = 30,
        window_seconds: int = 60,
        max_tracked_clients: int = 10000,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_tracked_clients = max_tracked_clients
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()
        self._last_cleanup = time.time()

    def _cleanup_expired(self, now: float | None = None) -> None:
        """Evict clients with no timestamps in the active window to prevent memory leaks."""
        current_time = now if now is not None else time.time()
        window_start = current_time - self.window_seconds
        expired_keys = [k for k, timestamps in self._requests.items() if not timestamps or timestamps[-1] <= window_start]
        for k in expired_keys:
            del self._requests[k]
        self._last_cleanup = current_time

    def check(self, key: str) -> tuple[bool, int, float]:
        """Check if request for given key is allowed.

        Returns:
            tuple[bool, int, float]:
                - allowed: True if request is under limit, False otherwise
                - remaining: Number of remaining requests in the window
                - retry_after: Seconds to wait before retrying (0.0 if allowed)
        """
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            # Periodic background cleanup every 30 seconds or if tracking table exceeds threshold
            if (now - self._last_cleanup > 30.0) or (len(self._requests) > self.max_tracked_clients):
                self._cleanup_expired(now)

            # Filter out timestamps outside the active window for this specific client
            valid_timestamps = [t for t in self._requests[key] if t > window_start]

            if len(valid_timestamps) >= self.max_requests:
                earliest = valid_timestamps[0]
                retry_after = max(0.0, round(earliest + self.window_seconds - now, 2))
                self._requests[key] = valid_timestamps
                return False, 0, retry_after

            # Record this request
            valid_timestamps.append(now)
            self._requests[key] = valid_timestamps
            remaining = self.max_requests - len(valid_timestamps)
            return True, remaining, 0.0

    def reset(self, key: str | None = None) -> None:
        """Reset the rate limiter state, useful for tests."""
        with self._lock:
            if key is not None:
                self._requests.pop(key, None)
            else:
                self._requests.clear()
