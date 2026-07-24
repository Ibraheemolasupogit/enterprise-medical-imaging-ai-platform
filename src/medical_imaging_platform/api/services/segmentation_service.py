"""Segmentation API service."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from medical_imaging_platform.api.errors import APIError
from medical_imaging_platform.api.models import DISCLAIMER, APIConfig, SegmentationPredictRequest
from medical_imaging_platform.api.security import (
    array_from_payload,
    load_npy,
    public_path,
    resolve_allowed_path,
    safe_output_child,
)
from medical_imaging_platform.segmentation.inference import predict_probability
from medical_imaging_platform.segmentation.postprocessing import postprocess_probability_map
from medical_imaging_platform.synthetic.manifest import sha256_file
from medical_imaging_platform.utils.config import load_segmentation_config


def predict_segmentation(
    request: SegmentationPredictRequest, config: APIConfig, request_id: str
) -> dict[str, object]:
    if config.segmentation_checkpoint is None:
        raise APIError(
            "API-NOTREADY-503", "Segmentation checkpoint is not configured.", status_code=503
        )
    image = (
        load_npy(resolve_allowed_path(request.input_path, config.allowed_input_roots), config)
        if request.input_path is not None
        else array_from_payload(request.array, config)  # type: ignore[arg-type]
    )
    segmentation_config = load_segmentation_config(Path("config/segmentation.yaml"))
    threshold = request.threshold
    if threshold is not None and not config.allow_threshold_override:
        raise APIError("API-AUTH-403", "Threshold override is disabled by policy.", status_code=403)
    start = time.perf_counter()
    try:
        probability, _ = predict_probability(
            image, config.segmentation_checkpoint, segmentation_config
        )
        mask, warnings, counts = postprocess_probability_map(
            probability, config=segmentation_config, threshold=threshold
        )
    except ValueError as exc:
        raise APIError(
            "API-VALID-422",
            "Segmentation inference input is invalid.",
            status_code=422,
        ) from exc
    except RuntimeError as exc:
        raise APIError(
            "API-INTEGRITY-409",
            "Segmentation checkpoint is incompatible with the configured model.",
            status_code=409,
        ) from exc
    output_paths: dict[str, str] = {}
    if request.persist_output:
        output_dir = safe_output_child(config.output_directory, f"{request_id}-segmentation")
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_numpy(output_dir / "predicted_mask.npy", mask.astype(np.uint8))
        output_paths["predicted_mask"] = public_path(output_dir / "predicted_mask.npy")
    spacing = np.asarray(request.spacing_mm, dtype=float)
    return {
        "request_id": request_id,
        "status": "completed",
        "probability_summary": {
            "min": float(np.min(probability)),
            "mean": float(np.mean(probability)),
            "max": float(np.max(probability)),
        },
        "predicted_voxel_count": int(np.count_nonzero(mask)),
        "predicted_volume_mm3": float(np.count_nonzero(mask) * np.prod(spacing)),
        "quality_status": "PASS" if not warnings else "PASS_WITH_WARNINGS",
        "quality_findings": warnings,
        "checkpoint_checksum": sha256_file(config.segmentation_checkpoint),
        "output_paths": output_paths,
        "duration_ms": float((time.perf_counter() - start) * 1000),
        "disclaimer": DISCLAIMER,
        "postprocessing": counts,
    }


def _write_numpy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.replace(path)
