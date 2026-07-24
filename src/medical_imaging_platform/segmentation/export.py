"""Experiment export and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from medical_imaging_platform.segmentation.report import render_segmentation_report
from medical_imaging_platform.synthetic.manifest import sha256_file
from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class SegmentationOutputError(MedicalImagingPlatformError):
    """Raised when segmentation evidence cannot be validated."""


def write_experiment_evidence(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    """Write deterministic experiment JSON and Markdown evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "experiment_config": output_dir / "experiment_config.json",
        "training_history": output_dir / "training_history.json",
        "evaluation": output_dir / "evaluation.json",
        "model_manifest": output_dir / "model_manifest.json",
        "report": output_dir / "segmentation_report.md",
    }
    _write_json(files["experiment_config"], payload["experiment_config"])
    _write_json(files["training_history"], {"history": payload["training_history"]})
    _write_json(files["evaluation"], payload["evaluation"])
    _write_json(files["model_manifest"], payload)
    _write_text(files["report"], render_segmentation_report(payload))
    return {key: sha256_file(path) for key, path in files.items()}


def validate_segmentation_experiment(output_dir: Path) -> dict[str, Any]:
    """Validate evidence files and recorded checksums."""
    manifest_path = output_dir / "model_manifest.json"
    if not manifest_path.exists():
        raise SegmentationOutputError("Model manifest is missing.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SegmentationOutputError(f"Model manifest is invalid JSON: {exc}") from exc
    for key in ("best_model", "last_model", "report"):
        path = output_dir / manifest["paths"][key]
        if not path.exists():
            raise SegmentationOutputError(f"Missing experiment artefact: {key}")
        expected = manifest["checksums"].get(key)
        if expected is not None and sha256_file(path) != expected:
            raise SegmentationOutputError(f"Checksum mismatch for {key}")
    return cast(dict[str, Any], manifest)


def inspect_segmentation_experiment(output_dir: Path) -> dict[str, Any]:
    """Return compact experiment summary."""
    manifest = validate_segmentation_experiment(output_dir)
    return {
        "experiment_id": manifest["experiment_id"],
        "status": manifest["status"],
        "dataset_id": manifest["dataset_id"],
        "best_epoch": manifest["best_epoch"],
        "best_validation_dice": manifest["best_validation_dice"],
        "model_parameter_count": manifest["model_parameter_count"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
