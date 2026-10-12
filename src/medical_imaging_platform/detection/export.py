"""Detection experiment export and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from medical_imaging_platform.detection.report import render_detection_report
from medical_imaging_platform.synthetic.manifest import sha256_file
from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class DetectionOutputError(MedicalImagingPlatformError):
    """Raised when detection experiment outputs fail validation."""


def write_experiment_evidence(output_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "experiment_config": output_dir / "experiment_config.json",
        "training_history": output_dir / "training_history.json",
        "evaluation": output_dir / "detection_evaluation.json",
        "predictions": output_dir / "detection_predictions.json",
        "model_manifest": output_dir / "detection_model_manifest.json",
        "report": output_dir / "detection_report.md",
    }
    _write_json(files["experiment_config"], payload["experiment_config"])
    _write_json(files["training_history"], {"history": payload["training_history"]})
    _write_json(files["evaluation"], payload["evaluation"])
    _write_json(files["predictions"], {"predictions": payload["predictions"]})
    _write_json(files["model_manifest"], payload)
    _write_text(files["report"], render_detection_report(payload))
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    checksums = {key: sha256_file(path) for key, path in files.items()}
    _write_json(
        evidence_dir / "checksum_manifest.json",
        {
            "algorithm": "sha256",
            "files": {
                str(path.relative_to(output_dir)): sha256_file(path) for path in files.values()
            },
        },
    )
    checksums["checksum_manifest"] = sha256_file(evidence_dir / "checksum_manifest.json")
    return checksums


def validate_detection_experiment(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "detection_model_manifest.json"
    if not manifest_path.exists():
        raise DetectionOutputError("Detection model manifest is missing.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DetectionOutputError(f"Invalid detection model manifest: {exc}") from exc
    for key in (
        "best_model",
        "last_model",
        "experiment_config",
        "training_history",
        "evaluation",
        "predictions",
        "report",
        "checksum_manifest",
    ):
        path = output_dir / manifest["paths"][key]
        if not path.exists():
            raise DetectionOutputError(f"Missing detection artefact: {key}")
        expected = manifest["checksums"].get(key)
        if key != "checksum_manifest" and expected is not None and sha256_file(path) != expected:
            raise DetectionOutputError(f"Checksum mismatch for {key}")
    return cast(dict[str, Any], manifest)


def inspect_detection_experiment(output_dir: Path) -> dict[str, Any]:
    manifest = validate_detection_experiment(output_dir)
    return {
        "experiment_id": manifest["experiment_id"],
        "status": manifest["status"],
        "dataset_id": manifest["dataset_id"],
        "model_family": manifest["model_family"],
        "test_map": manifest["evaluation"]["test"]["metrics"]["map"],
        "test_ap50": manifest["evaluation"]["test"]["metrics"]["ap50"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
