"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from medical_imaging_platform.api.errors import (
    APIError,
    api_error_handler,
    error_payload,
    request_id,
    unhandled_error_handler,
)
from medical_imaging_platform.api.middleware import RequestContextMiddleware
from medical_imaging_platform.api.models import DISCLAIMER
from medical_imaging_platform.api.observability import MetricsRegistry, structured_log_event
from medical_imaging_platform.api.routes import (
    classification,
    health,
    longitudinal,
    review,
    segmentation,
)
from medical_imaging_platform.utils.config import load_api_config


def create_app(config_path: Path | None = None) -> FastAPI:
    """Create the local research-only API app."""
    config = load_api_config(config_path or Path("config/api.yaml"))
    app = FastAPI(
        title=config.service_name,
        version=config.service_version,
        description=DISCLAIMER,
        docs_url="/docs" if config.enable_docs else None,
        openapi_url="/openapi.json" if config.enable_openapi else None,
    )
    app.state.api_config = config
    app.state.metrics_registry = MetricsRegistry()
    app.state.startup_status = {
        "status": "initialised",
        "model_configured": any(
            [
                config.segmentation_checkpoint,
                config.classification_checkpoint,
                config.classification_calibration,
                config.classification_threshold_policy,
            ]
        ),
        "event": structured_log_event(
            service="api",
            event_type="startup",
            severity="INFO",
            details={"environment": config.environment, "raw_payload": "not logged"},
        ),
    }
    app.add_middleware(RequestContextMiddleware, config=config)
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.include_router(health.router)
    app.include_router(segmentation.router)
    app.include_router(classification.router)
    app.include_router(longitudinal.router)
    app.include_router(review.router)
    return app


async def _validation_handler(request: Request, exc: RequestValidationError) -> object:
    error = APIError("API-VALID-422", "Request validation failed.", status_code=422)
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=422, content=error_payload(error, request_id(request)))
