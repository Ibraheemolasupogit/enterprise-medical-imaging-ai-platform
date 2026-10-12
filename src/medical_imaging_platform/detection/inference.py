"""Detection inference for prepared 2D slices."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from medical_imaging_platform.detection.checkpoint import load_state_dict
from medical_imaging_platform.detection.model_factory import build_detector, predictions_to_common
from medical_imaging_platform.detection.models import DetectionConfig
from medical_imaging_platform.synthetic.manifest import sha256_file


def detect_slice(
    input_slice: Path,
    *,
    checkpoint_path: Path,
    output_dir: Path,
    config: DetectionConfig,
    overwrite: bool,
    source_provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run detection on one prepared 2D `.npy` slice."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Detection inference output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image = np.load(input_slice).astype(np.float32)
    prediction, duration = predict_array(image, checkpoint_path, config)
    metadata = {
        "status": "completed",
        "model_family": config.model_family,
        "input_shape": list(image.shape),
        "prediction": prediction,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_checksum": sha256_file(checkpoint_path),
        "source_provenance": source_provenance
        or {"input_slice_checksum": sha256_file(input_slice)},
        "duration_ms": duration * 1000.0,
        "warnings": [],
        "disclaimer": (
            "Synthetic/de-identified engineering detection output only; not diagnosis, triage, "
            "RECIST, patient-management, or medical-device evidence."
        ),
    }
    _write_json(output_dir / "detection_inference_metadata.json", metadata)
    return metadata


def predict_array(
    image: np.ndarray, checkpoint_path: Path, config: DetectionConfig
) -> tuple[dict[str, Any], float]:
    if image.shape != config.image_shape or image.ndim != 2:
        raise ValueError("Detection input slice must match configured 2D image_shape.")
    if not np.all(np.isfinite(image)):
        raise ValueError("Detection input slice contains non-finite values.")
    model = build_detector(config)
    model.load_state_dict(load_state_dict(checkpoint_path), strict=False)
    model.eval()
    tensor = torch.as_tensor(image[None], dtype=torch.float32)
    start = time.perf_counter()
    with torch.no_grad():
        raw = model.predict([tensor]) if hasattr(model, "predict") else model([tensor])
    duration = float(time.perf_counter() - start)
    return predictions_to_common(cast(list[dict[str, torch.Tensor]], raw))[0], duration


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
