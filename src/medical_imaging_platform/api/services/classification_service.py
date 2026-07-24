"""Classification API service."""

from __future__ import annotations

import json
import time
from pathlib import Path

from medical_imaging_platform.api.errors import APIError
from medical_imaging_platform.api.models import DISCLAIMER, APIConfig, ClassificationPredictRequest
from medical_imaging_platform.api.security import array_from_payload, load_npy, resolve_allowed_path
from medical_imaging_platform.classification.inference import classify_array
from medical_imaging_platform.synthetic.manifest import sha256_file
from medical_imaging_platform.utils.config import load_classification_config


def predict_classification(
    request: ClassificationPredictRequest, config: APIConfig, request_id: str
) -> dict[str, object]:
    if (
        config.classification_checkpoint is None
        or config.classification_calibration is None
        or config.classification_threshold_policy is None
    ):
        raise APIError(
            "API-NOTREADY-503", "Classification artefacts are not configured.", status_code=503
        )
    image = (
        load_npy(resolve_allowed_path(request.input_path, config.allowed_input_roots), config)
        if request.input_path is not None
        else array_from_payload(request.array, config)  # type: ignore[arg-type]
    )
    classification_config = load_classification_config(Path("config/classification.yaml"))
    calibration = json.loads(config.classification_calibration.read_text(encoding="utf-8"))
    threshold_policy = json.loads(
        config.classification_threshold_policy.read_text(encoding="utf-8")
    )
    start = time.perf_counter()
    try:
        prediction, _ = classify_array(
            image,
            config.classification_checkpoint,
            calibration,
            threshold_policy,
            classification_config,
        )
    except ValueError as exc:
        raise APIError(
            "API-VALID-422",
            "Classification inference input is invalid.",
            status_code=422,
        ) from exc
    except RuntimeError as exc:
        raise APIError(
            "API-INTEGRITY-409",
            "Classification checkpoint is incompatible with the configured model.",
            status_code=409,
        ) from exc
    return {
        "request_id": request_id,
        "raw_probability": prediction["raw_probability"],
        "calibrated_probability": prediction["calibrated_probability"],
        "threshold": prediction["threshold"],
        "engineering_label": prediction["final_label"],
        "abstained": prediction["abstained"],
        "abstention_reason": prediction["abstention_reason"],
        "quality_status": "PASS" if not prediction["abstained"] else "PASS_WITH_WARNINGS",
        "checkpoint_checksum": sha256_file(config.classification_checkpoint),
        "calibration_checksum": sha256_file(config.classification_calibration),
        "threshold_policy_checksum": sha256_file(config.classification_threshold_policy),
        "duration_ms": float((time.perf_counter() - start) * 1000),
        "disclaimer": DISCLAIMER,
    }
