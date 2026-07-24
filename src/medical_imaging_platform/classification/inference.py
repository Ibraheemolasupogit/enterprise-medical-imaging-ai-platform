"""Classification inference and abstention."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from medical_imaging_platform.classification.calibration import apply_calibration, sigmoid
from medical_imaging_platform.classification.checkpoint import load_state_dict
from medical_imaging_platform.classification.model_factory import build_classifier
from medical_imaging_platform.classification.models import ClassificationConfig
from medical_imaging_platform.classification.pipeline import resolve_device, set_reproducibility
from medical_imaging_platform.synthetic.manifest import sha256_file


def classify_array(
    image: np.ndarray,
    checkpoint_path: Path,
    calibration: dict[str, object],
    threshold_policy: dict[str, object],
    config: ClassificationConfig,
) -> tuple[dict[str, object], float]:
    """Classify one ROI array."""
    if image.shape != config.input_shape or not np.all(np.isfinite(image)):
        raise ValueError("Classification input must match configured shape and be finite.")
    set_reproducibility(config.random_seed)
    device = resolve_device(config.device)
    model = build_classifier(config).to(device)
    model.load_state_dict(load_state_dict(checkpoint_path))
    model.eval()
    tensor = torch.as_tensor(
        image[None, None].astype(np.float32), dtype=torch.float32, device=device
    )
    start = time.perf_counter()
    with torch.no_grad():
        logit = float(model(tensor).cpu().numpy()[0])
    duration = float(time.perf_counter() - start)
    raw_probability = float(sigmoid(np.asarray([logit]))[0])
    calibrated = apply_calibration(logit, calibration)
    threshold = _threshold_value(threshold_policy)
    decision = apply_abstention(calibrated, threshold, config)
    return {
        "raw_logit": logit,
        "raw_probability": raw_probability,
        "calibrated_probability": calibrated,
        "threshold": threshold,
        **decision,
    }, duration


def classify_volume(
    input_volume: Path,
    *,
    checkpoint_path: Path,
    calibration_path: Path,
    threshold_policy_path: Path,
    output_dir: Path,
    config: ClassificationConfig,
    overwrite: bool,
) -> dict[str, object]:
    """Run inference for one explicit ROI volume and write evidence."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Classification inference output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    threshold_policy = json.loads(threshold_policy_path.read_text(encoding="utf-8"))
    prediction, duration = classify_array(
        np.load(input_volume).astype(np.float32),
        checkpoint_path,
        calibration,
        threshold_policy,
        config,
    )
    metadata = {
        "input_volume": str(input_volume),
        "checkpoint_checksum": sha256_file(checkpoint_path),
        "calibration_checksum": sha256_file(calibration_path),
        "threshold_policy_checksum": sha256_file(threshold_policy_path),
        "inference_duration_seconds": duration,
        "source_provenance": {"input_path": str(input_volume)},
    }
    _write_json(output_dir / "prediction.json", prediction)
    _write_json(
        output_dir / "probability.json",
        {
            "raw_probability": prediction["raw_probability"],
            "calibrated_probability": prediction["calibrated_probability"],
        },
    )
    _write_json(output_dir / "inference_metadata.json", metadata)
    return {"prediction": prediction, "metadata": metadata}


def apply_abstention(
    probability: float, threshold: float, config: ClassificationConfig
) -> dict[str, object]:
    """Apply confidence-based indeterminate interval."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be within [0, 1].")
    if config.abstention_lower <= probability <= config.abstention_upper:
        return {
            "final_label": "indeterminate",
            "abstained": True,
            "abstention_reason": "Probability inside configured indeterminate interval.",
            "abstention_range": [config.abstention_lower, config.abstention_upper],
        }
    label = (
        "synthetic_lesion_present" if probability >= threshold else "no_visible_synthetic_lesion"
    )
    return {
        "final_label": label,
        "abstained": False,
        "abstention_reason": None,
        "abstention_range": [config.abstention_lower, config.abstention_upper],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _threshold_value(threshold_policy: dict[str, object]) -> float:
    value = threshold_policy.get("selected_threshold")
    if isinstance(value, int | float | str):
        return float(value)
    raise ValueError("Threshold policy is missing a numeric selected_threshold.")
