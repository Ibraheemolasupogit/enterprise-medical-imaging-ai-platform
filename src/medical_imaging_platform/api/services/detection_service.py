"""Detection inference service for the governed local API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

import numpy as np

from medical_imaging_platform.api.errors import APIError
from medical_imaging_platform.api.models import DISCLAIMER, APIConfig, DetectionPredictRequest
from medical_imaging_platform.api.security import resolve_allowed_path
from medical_imaging_platform.detection.inference import predict_array
from medical_imaging_platform.synthetic.manifest import sha256_file
from medical_imaging_platform.utils.config import load_detection_config


def predict_detection(
    payload: DetectionPredictRequest, config: APIConfig, request_id: str
) -> dict[str, object]:
    if config.detection_checkpoint is None:
        raise APIError("API-MODEL-503", "Detection checkpoint is not configured.", status_code=503)
    checkpoint = Path(config.detection_checkpoint)
    if not checkpoint.exists():
        raise APIError("API-MODEL-409", "Detection checkpoint is missing.", status_code=409)
    detection_config = load_detection_config(config.detection_config)
    image = _load_image(payload, config)
    start = time.perf_counter()
    try:
        prediction, duration = predict_array(image, checkpoint, detection_config)
    except Exception as exc:
        raise APIError("API-MODEL-409", "Detection inference failed.", status_code=409) from exc
    total_duration = float(time.perf_counter() - start)
    output_paths: dict[str, str] = {}
    if payload.persist_output:
        output_dir = config.output_directory / "detection" / request_id
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "detection_prediction.json"
        metadata_path.write_text(
            json.dumps(
                {"prediction": prediction, "request_id": request_id}, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        output_paths["prediction"] = str(metadata_path.relative_to(config.output_directory))
    return {
        "status": "completed",
        "request_id": request_id,
        "model_family": detection_config.model_family,
        "boxes": prediction["boxes"],
        "labels": prediction["labels"],
        "scores": prediction["scores"],
        "box_count": len(prediction["boxes"]),
        "checkpoint_checksum": sha256_file(checkpoint),
        "source_provenance": payload.source_provenance,
        "output_paths": output_paths,
        "duration_ms": total_duration * 1000.0,
        "inference_duration_ms": duration * 1000.0,
        "warnings": [
            "2D synthetic axial boxes are engineering evidence only and do not represent "
            "full 3D extent."
        ],
        "disclaimer": DISCLAIMER,
    }


def _load_image(payload: DetectionPredictRequest, config: APIConfig) -> np.ndarray:
    if payload.input_path is not None:
        path = resolve_allowed_path(payload.input_path, config.allowed_input_roots)
        if path.stat().st_size > config.maximum_array_bytes:
            raise APIError(
                "API-SIZE-413", "Array file exceeds configured size limit.", status_code=413
            )
        data = np.load(path, allow_pickle=False).astype(np.float32)
    else:
        assert payload.array is not None
        if len(payload.array.values) != int(np.prod(payload.array.shape)):
            raise APIError(
                "API-VALID-422", "Array payload length does not match shape.", status_code=422
            )
        if len(payload.array.values) * 8 > config.maximum_array_bytes:
            raise APIError(
                "API-SIZE-413", "Array payload exceeds configured size limit.", status_code=413
            )
        data = np.asarray(payload.array.values, dtype=np.float32).reshape(payload.array.shape)
    if data.ndim != 2 or not np.all(np.isfinite(data)):
        raise APIError(
            "API-VALID-422", "Detection input must be a finite 2D array.", status_code=422
        )
    return cast(np.ndarray, data)
