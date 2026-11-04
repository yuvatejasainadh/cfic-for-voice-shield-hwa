"""Unit tests for sliding-window rate limiter."""

import time
from app.core.rate_limiter import InMemoryRateLimiter


class TestRateLimiter:
    def test_allows_requests_under_limit(self):
        limiter = InMemoryRateLimiter(max_requests=5, window_seconds=10)
        for i in range(5):
            allowed, remaining, retry_after = limiter.check("client_a")
            assert allowed is True
            assert remaining == 5 - (i + 1)
            assert retry_after == 0.0

    def test_blocks_request_exceeding_limit(self):
        limiter = InMemoryRateLimiter(max_requests=3, window_seconds=10)
        for _ in range(3):
            allowed, _, _ = limiter.check("client_b")
            assert allowed is True

        # 4th request must be blocked
        allowed, remaining, retry_after = limiter.check("client_b")
        assert allowed is False
        assert remaining == 0
        assert retry_after > 0.0

    def test_independent_clients_do_not_interfere(self):
        limiter = InMemoryRateLimiter(max_requests=2, window_seconds=10)
        limiter.check("client_1")
        limiter.check("client_1")
        # client_1 is now exhausted
        assert limiter.check("client_1")[0] is False

        # client_2 should still be allowed
        allowed, remaining, _ = limiter.check("client_2")
        assert allowed is True
        assert remaining == 1

    def test_reset_functionality(self):
        limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
        limiter.check("client_reset")
        assert limiter.check("client_reset")[0] is False

        limiter.reset("client_reset")
        assert limiter.check("client_reset")[0] is True
