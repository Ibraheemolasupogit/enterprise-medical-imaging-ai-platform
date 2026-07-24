"""Longitudinal API service."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from medical_imaging_platform.api.errors import APIError
from medical_imaging_platform.api.models import DISCLAIMER, APIConfig, LongitudinalAnalyseRequest
from medical_imaging_platform.api.security import (
    array_from_payload,
    public_path,
    resolve_allowed_path,
)
from medical_imaging_platform.longitudinal.pipeline import (
    LongitudinalAnalysisError,
    analyse_longitudinal_pair,
)
from medical_imaging_platform.utils.config import load_longitudinal_config


def analyse_longitudinal(
    request: LongitudinalAnalyseRequest, config: APIConfig, request_id: str
) -> dict[str, object]:
    start = time.perf_counter()
    if request.previous_mask_path is not None and request.current_mask_path is not None:
        previous_path = resolve_allowed_path(request.previous_mask_path, config.allowed_input_roots)
        current_path = resolve_allowed_path(request.current_mask_path, config.allowed_input_roots)
    else:
        previous_path = _write_api_array(
            config.output_directory / f"{request_id}-previous.npy",
            array_from_payload(request.previous_array, config),  # type: ignore[arg-type]
        )
        current_path = _write_api_array(
            config.output_directory / f"{request_id}-current.npy",
            array_from_payload(request.current_array, config),  # type: ignore[arg-type]
        )
    upstream = {
        "registration": request.registration_status,
        "segmentation": request.segmentation_status,
        "classification": request.classification_status,
        "classification_abstention": request.classification_abstention_status,
    }
    try:
        payload = analyse_longitudinal_pair(
            previous_mask_path=previous_path,
            current_mask_path=current_path,
            previous_spacing_mm=request.previous_spacing_mm,
            current_spacing_mm=request.current_spacing_mm,
            case_id=request.case_id,
            research_subject_id=request.research_subject_id,
            side=request.side,
            previous_timepoint=request.previous_timepoint,
            current_timepoint=request.current_timepoint,
            output_root=config.output_directory,
            config=load_longitudinal_config(config.longitudinal_config),
            registration_run_id=request.registration_run_id,
            upstream_quality_statuses=upstream,
            overwrite=True,
        )
    except LongitudinalAnalysisError as exc:
        raise APIError(
            "API-VALID-422",
            "Longitudinal analysis input is invalid.",
            status_code=422,
        ) from exc
    analysis_dir = config.output_directory / str(payload["analysis_id"])
    return {
        "request_id": request_id,
        "measurements": payload["lesion_measurements"],
        "match_summary": payload["lesion_matches"],
        "change_metrics": payload["longitudinal_changes"],
        "engineering_label": payload["summary"]["engineering_labels"],
        "upstream_quality_propagation": upstream,
        "quality_findings": payload["quality_findings"],
        "evidence_path": public_path(analysis_dir) if request.persist_output else None,
        "duration_ms": float((time.perf_counter() - start) * 1000),
        "disclaimer": DISCLAIMER,
    }


def _write_api_array(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array.astype(np.uint8))
    tmp.replace(path)
    return path
