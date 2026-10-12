"""Detection prediction route."""
# mypy: disable-error-code="misc"

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from medical_imaging_platform.api.dependencies import get_api_config
from medical_imaging_platform.api.models import APIConfig, DetectionPredictRequest
from medical_imaging_platform.api.services.detection_service import predict_detection

router = APIRouter(prefix="/v1/detection", tags=["detection"])


@router.post("/predict")
def predict(
    payload: DetectionPredictRequest,
    request: Request,
    config: Annotated[APIConfig, Depends(get_api_config)],
) -> dict[str, object]:
    return predict_detection(payload, config, payload.request_id or request.state.request_id)
