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
    """Sliding-window in-memory rate limiter."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

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
            # Filter out timestamps outside the active window
            valid_timestamps = [t for t in self._requests[key] if t > window_start]
            self._requests[key] = valid_timestamps

            if len(valid_timestamps) >= self.max_requests:
                earliest = valid_timestamps[0]
                retry_after = max(0.0, round(earliest + self.window_seconds - now, 2))
                return False, 0, retry_after

            # Record this request
            self._requests[key].append(now)
            remaining = self.max_requests - len(self._requests[key])
            return True, remaining, 0.0

    def reset(self, key: str | None = None) -> None:
        """Reset the rate limiter state, useful for tests."""
        with self._lock:
            if key is not None:
                self._requests.pop(key, None)
            else:
                self._requests.clear()
