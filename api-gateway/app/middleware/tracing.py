"""Request ID tracing middleware for CFIC API Gateway."""

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Ensure every request has an X-Request-ID header attached to both request state and response."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
