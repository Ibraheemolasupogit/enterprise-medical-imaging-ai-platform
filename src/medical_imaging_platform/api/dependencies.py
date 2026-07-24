"""FastAPI dependency helpers."""

from __future__ import annotations

from fastapi import Request

from medical_imaging_platform.api.models import APIConfig


def get_api_config(request: Request) -> APIConfig:
    return request.app.state.api_config  # type: ignore[no-any-return]
