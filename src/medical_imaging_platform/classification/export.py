"""Classification experiment export and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from medical_imaging_platform.classification.report import render_classification_report
from medical_imaging_platform.synthetic.manifest import sha256_file
from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class ClassificationOutputError(MedicalImagingPlatformError):
    """Raised when classification outputs fail validation."""


def write_experiment_evidence(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "calibration": output_dir / "calibration.json",
        "threshold_policy": output_dir / "threshold_policy.json",
        "experiment_config": output_dir / "experiment_config.json",
        "training_history": output_dir / "training_history.json",
        "evaluation": output_dir / "evaluation.json",
        "predictions": output_dir / "predictions.json",
        "model_manifest": output_dir / "model_manifest.json",
        "report": output_dir / "classification_report.md",
    }
    _write_json(files["calibration"], payload["calibration"])
    _write_json(files["threshold_policy"], payload["threshold_policy"])
    _write_json(files["experiment_config"], payload["experiment_config"])
    _write_json(files["training_history"], {"history": payload["training_history"]})
    _write_json(files["evaluation"], payload["evaluation"])
    _write_json(files["predictions"], {"predictions": payload["predictions"]})
    _write_json(files["model_manifest"], payload)
    _write_text(files["report"], render_classification_report(payload))
    return {key: sha256_file(path) for key, path in files.items()}


def validate_classification_experiment(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "model_manifest.json"
    if not manifest_path.exists():
        raise ClassificationOutputError("Classification model manifest is missing.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ClassificationOutputError(f"Invalid classification model manifest: {exc}") from exc
    for key in ("best_model", "last_model", "calibration", "threshold_policy", "report"):
        path = output_dir / manifest["paths"][key]
        if not path.exists():
            raise ClassificationOutputError(f"Missing classification artefact: {key}")
        expected = manifest["checksums"].get(key)
        if expected is not None and sha256_file(path) != expected:
            raise ClassificationOutputError(f"Checksum mismatch for {key}")
    return cast(dict[str, Any], manifest)


def inspect_classification_experiment(output_dir: Path) -> dict[str, Any]:
    manifest = validate_classification_experiment(output_dir)
    return {
        "experiment_id": manifest["experiment_id"],
        "status": manifest["status"],
        "dataset_id": manifest["dataset_id"],
        "best_epoch": manifest["best_epoch"],
        "validation_auroc": manifest["evaluation"]["validation"]["metrics"]["auroc"],
        "test_auroc": manifest["evaluation"]["test"]["metrics"]["auroc"],
        "selected_threshold": manifest["threshold_policy"]["selected_threshold"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
