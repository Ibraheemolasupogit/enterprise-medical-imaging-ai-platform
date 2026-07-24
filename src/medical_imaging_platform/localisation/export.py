"""Localisation output export and validation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from medical_imaging_platform.localisation.models import LocalisationOutputPaths, LocalisationResult
from medical_imaging_platform.localisation.report import render_localisation_report
from medical_imaging_platform.synthetic.manifest import sha256_file
from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class LocalisationOutputError(MedicalImagingPlatformError):
    """Raised when localisation output cannot be written or validated."""


def build_output_paths(output_root: Path, run_id: str) -> LocalisationOutputPaths:
    root = output_root.resolve()
    output_dir = (root / run_id).resolve()
    if os.path.commonpath([str(root), str(output_dir)]) != str(root):
        raise LocalisationOutputError("Localisation output path escapes configured output root.")
    return LocalisationOutputPaths(
        output_dir=str(output_dir),
        left_roi=str(output_dir / "left_roi.npy"),
        right_roi=str(output_dir / "right_roi.npy"),
        localisation_json=str(output_dir / "localisation.json"),
        report=str(output_dir / "localisation_report.md"),
        left_overlay=str(output_dir / "left_overlay.npy"),
        right_overlay=str(output_dir / "right_overlay.npy"),
    )


def write_localisation_outputs(
    left_roi: np.ndarray,
    right_roi: np.ndarray,
    left_overlay: np.ndarray,
    right_overlay: np.ndarray,
    result: LocalisationResult,
    *,
    overwrite: bool,
) -> LocalisationResult:
    paths = _path_map(result.output_paths)
    output_dir = Path(result.output_paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(path.exists() for path in paths.values()):
        raise LocalisationOutputError(f"Localisation output already exists in {output_dir}")
    _write_numpy(paths["left_roi"], left_roi)
    _write_numpy(paths["right_roi"], right_roi)
    _write_numpy(paths["left_overlay"], left_overlay)
    _write_numpy(paths["right_overlay"], right_overlay)
    checksums = {
        key: sha256_file(path)
        for key, path in paths.items()
        if key not in {"localisation_json", "report"}
    }
    result = result.model_copy(update={"checksums": checksums})
    _write_text(
        paths["localisation_json"],
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    _write_text(paths["report"], render_localisation_report(result))
    result = result.model_copy(
        update={"checksums": {**result.checksums, "report": sha256_file(paths["report"])}}
    )
    _write_text(
        paths["localisation_json"],
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    validate_localisation_output(output_dir)
    return result


def validate_localisation_output(output_dir: Path) -> LocalisationResult:
    metadata_path = output_dir / "localisation.json"
    if not metadata_path.exists():
        raise LocalisationOutputError("Localisation JSON is missing.")
    try:
        result = LocalisationResult.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        left = np.load(output_dir / "left_roi.npy")
        right = np.load(output_dir / "right_roi.npy")
    except Exception as exc:
        raise LocalisationOutputError(f"Localisation output cannot be loaded: {exc}") from exc
    if tuple(left.shape) != result.left.roi_shape or tuple(right.shape) != result.right.roi_shape:
        raise LocalisationOutputError("ROI shapes do not match localisation metadata.")
    for key, checksum in result.checksums.items():
        if sha256_file(_path_map(result.output_paths)[key]) != checksum:
            raise LocalisationOutputError(f"Checksum mismatch for localisation output: {key}")
    return result


def inspect_localisation_output(output_dir: Path) -> dict[str, object]:
    result = validate_localisation_output(output_dir)
    return {
        "localisation_run_id": result.localisation_run_id,
        "status": result.overall_status,
        "source_run_id": result.source.source_run_id,
        "left_centre": list(result.left.predicted_centre_voxel),
        "right_centre": list(result.right.predicted_centre_voxel),
        "ground_truth_available": result.ground_truth_available,
        "warnings": result.warnings,
    }


def _path_map(paths: LocalisationOutputPaths) -> dict[str, Path]:
    return {
        "left_roi": Path(paths.left_roi),
        "right_roi": Path(paths.right_roi),
        "localisation_json": Path(paths.localisation_json),
        "report": Path(paths.report),
        "left_overlay": Path(paths.left_overlay),
        "right_overlay": Path(paths.right_overlay),
    }


def _write_numpy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.replace(path)


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
