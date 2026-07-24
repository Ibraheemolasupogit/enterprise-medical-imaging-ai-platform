"""Classification prediction route."""
# mypy: disable-error-code="misc"

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from medical_imaging_platform.api.dependencies import get_api_config
from medical_imaging_platform.api.models import APIConfig, ClassificationPredictRequest
from medical_imaging_platform.api.services.classification_service import predict_classification

router = APIRouter(prefix="/v1/classification", tags=["classification"])


@router.post("/predict")
def predict(
    payload: ClassificationPredictRequest,
    request: Request,
    config: Annotated[APIConfig, Depends(get_api_config)],
) -> dict[str, object]:
    return predict_classification(payload, config, payload.request_id or request.state.request_id)
