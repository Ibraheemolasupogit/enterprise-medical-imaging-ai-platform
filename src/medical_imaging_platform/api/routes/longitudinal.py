"""Longitudinal analysis route."""
# mypy: disable-error-code="misc"

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from medical_imaging_platform.api.dependencies import get_api_config
from medical_imaging_platform.api.models import APIConfig, LongitudinalAnalyseRequest
from medical_imaging_platform.api.services.longitudinal_service import analyse_longitudinal

router = APIRouter(prefix="/v1/longitudinal", tags=["longitudinal"])


@router.post("/analyse")
def analyse(
    payload: LongitudinalAnalyseRequest,
    request: Request,
    config: Annotated[APIConfig, Depends(get_api_config)],
) -> dict[str, object]:
    return analyse_longitudinal(payload, config, payload.request_id or request.state.request_id)
