"""API middleware for request IDs, limits, logging, and security headers."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from medical_imaging_platform.api.errors import APIError, error_payload
from medical_imaging_platform.api.models import APIConfig
from medical_imaging_platform.api.observability import structured_log_event


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
        latency = time.perf_counter() - start
        registry = getattr(request.app.state, "metrics_registry", None)
        if registry is not None:
            registry.record_request(request.method, request.url.path, response.status_code, latency)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Correlation-ID"] = request.headers.get("X-Correlation-ID", rid)
        response.headers["X-Process-Time-Ms"] = f"{latency * 1000:.3f}"
        request.app.state.last_structured_log = structured_log_event(
            service="api",
            event_type="request_completed",
            request_id=rid,
            correlation_id=response.headers["X-Correlation-ID"],
            details={
                "method": request.method,
                "route": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(latency * 1000, 3),
            },
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'none'"
        return response
