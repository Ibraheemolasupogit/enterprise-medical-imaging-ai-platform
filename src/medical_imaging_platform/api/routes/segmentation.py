"""Segmentation prediction route."""
# mypy: disable-error-code="misc"

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from medical_imaging_platform.api.dependencies import get_api_config
from medical_imaging_platform.api.models import APIConfig, SegmentationPredictRequest
from medical_imaging_platform.api.services.segmentation_service import predict_segmentation

router = APIRouter(prefix="/v1/segmentation", tags=["segmentation"])


@router.post("/predict")
def predict(
    payload: SegmentationPredictRequest,
    request: Request,
    config: Annotated[APIConfig, Depends(get_api_config)],
) -> dict[str, object]:
    return predict_segmentation(payload, config, payload.request_id or request.state.request_id)
