"""API middleware for request IDs, limits, logging, and security headers."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from medical_imaging_platform.api.errors import APIError, error_payload
from medical_imaging_platform.api.models import APIConfig


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach request ID, enforce size, and emit security headers."""

    def __init__(self, app: object, config: APIConfig) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.config = config

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("X-Request-ID") or f"api-{uuid.uuid4().hex[:16]}"
        request.state.request_id = rid
        length = request.headers.get("content-length")
        if length is not None:
            try:
                request_bytes = int(length)
            except ValueError:
                error = APIError("API-REQ-400", "Invalid Content-Length header.", status_code=400)
                from fastapi.responses import JSONResponse

                return JSONResponse(status_code=400, content=error_payload(error, rid))
        else:
            request_bytes = 0
        if request_bytes > self.config.maximum_request_bytes:
            error = APIError("API-SIZE-413", "Request body is too large.", status_code=413)
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=413, content=error_payload(error, rid))
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.3f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'none'"
        return response
