"""Read-only review evidence routes."""
# mypy: disable-error-code="misc"

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from medical_imaging_platform.api.dependencies import get_api_config
from medical_imaging_platform.api.models import APIConfig
from medical_imaging_platform.api.services.review_service import (
    review_classification,
    review_longitudinal,
    review_segmentation,
)

router = APIRouter(prefix="/v1/review", tags=["review"])


@router.get("/segmentation/{experiment_id}")
def segmentation(
    experiment_id: str, config: Annotated[APIConfig, Depends(get_api_config)]
) -> dict[str, object]:
    return review_segmentation(experiment_id, config)


@router.get("/classification/{experiment_id}")
def classification(
    experiment_id: str, config: Annotated[APIConfig, Depends(get_api_config)]
) -> dict[str, object]:
    return review_classification(experiment_id, config)


@router.get("/longitudinal/{analysis_id}")
def longitudinal(
    analysis_id: str, config: Annotated[APIConfig, Depends(get_api_config)]
) -> dict[str, object]:
    return review_longitudinal(analysis_id, config)
