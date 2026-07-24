"""API error types and sanitised responses."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


class APIError(Exception):
    """Domain API error with deterministic code and status."""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        status_code: int,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def error_payload(error: APIError, rid: str) -> dict[str, Any]:
    return {
        "error_code": error.error_code,
        "message": error.message,
        "request_id": rid,
        "details": error.details,
    }


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code, content=error_payload(exc, request_id(request))
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    error = APIError("API-INTERNAL-500", "Internal API error.", status_code=500)
    return JSONResponse(status_code=500, content=error_payload(error, request_id(request)))
