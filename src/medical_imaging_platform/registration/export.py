"""Registration output export and validation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from medical_imaging_platform.registration.models import RegistrationOutputPaths, RegistrationResult
from medical_imaging_platform.registration.report import render_registration_report
from medical_imaging_platform.synthetic.manifest import sha256_file
from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class RegistrationOutputError(MedicalImagingPlatformError):
    """Raised when registration output cannot be written or validated."""


def build_output_paths(output_root: Path, registration_run_id: str) -> RegistrationOutputPaths:
    root = output_root.resolve()
    output_dir = (root / registration_run_id).resolve()
    if os.path.commonpath([str(root), str(output_dir)]) != str(root):
        raise RegistrationOutputError("Registration output path escapes configured output root.")
    return RegistrationOutputPaths(
        output_dir=str(output_dir),
        registered_moving_volume=str(output_dir / "registered_moving_volume.npy"),
        transform=str(output_dir / "transform.json"),
        metrics=str(output_dir / "metrics.json"),
        metadata=str(output_dir / "registration_metadata.json"),
        report=str(output_dir / "registration_report.md"),
        overlay_mid_axial=str(output_dir / "overlay_mid_axial.npy"),
        difference_mid_axial=str(output_dir / "difference_mid_axial.npy"),
        fixed_mid_axial=str(output_dir / "fixed_mid_axial.npy"),
        moving_mid_axial=str(output_dir / "moving_mid_axial.npy"),
        registered_mid_axial=str(output_dir / "registered_mid_axial.npy"),
    )


def write_registration_outputs(
    registered: np.ndarray,
    review_arrays: dict[str, np.ndarray],
    result: RegistrationResult,
    *,
    overwrite: bool,
) -> RegistrationResult:
    output_dir = Path(result.output_paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = _output_file_map(result.output_paths)
    if not overwrite and any(path.exists() for path in output_files.values()):
        raise RegistrationOutputError(f"Registration output already exists in {output_dir}")

    _write_numpy_atomic(output_files["registered_moving_volume"], registered)
    for name, array in review_arrays.items():
        _write_numpy_atomic(output_files[name], array)
    _write_text_atomic(
        output_files["transform"],
        json.dumps(result.transform.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        output_files["metrics"],
        json.dumps(
            {
                "before": result.metrics_before.model_dump(mode="json"),
                "after": result.metrics_after.model_dump(mode="json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    checksums = {
        key: sha256_file(path)
        for key, path in output_files.items()
        if key not in {"metadata", "report"}
    }
    result = result.model_copy(update={"checksums": checksums})
    _write_text_atomic(
        output_files["metadata"],
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(output_files["report"], render_registration_report(result))
    result = result.model_copy(
        update={"checksums": {**result.checksums, "report": sha256_file(output_files["report"])}}
    )
    _write_text_atomic(
        output_files["metadata"],
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    validate_registration_output(output_dir)
    return result


def validate_registration_output(output_dir: Path) -> RegistrationResult:
    metadata_path = output_dir / "registration_metadata.json"
    if not metadata_path.exists():
        raise RegistrationOutputError("Registration metadata is missing.")
    try:
        result = RegistrationResult.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        registered = np.load(output_dir / "registered_moving_volume.npy")
    except Exception as exc:
        raise RegistrationOutputError(f"Registration output cannot be loaded: {exc}") from exc
    if tuple(registered.shape) != result.fixed.volume_shape:
        raise RegistrationOutputError("Registered volume shape does not match fixed volume shape.")
    for key, checksum in result.checksums.items():
        path = _output_file_map(result.output_paths)[key]
        if sha256_file(path) != checksum:
            raise RegistrationOutputError(f"Checksum mismatch for registration output: {key}")
    return result


def inspect_registration_output(output_dir: Path) -> dict[str, object]:
    result = validate_registration_output(output_dir)
    registered = np.load(output_dir / "registered_moving_volume.npy")
    return {
        "registration_run_id": result.registration_run_id,
        "status": result.status,
        "mode": result.mode,
        "shape": list(registered.shape),
        "dtype": str(registered.dtype),
        "mse_before": result.metrics_before.mean_squared_error,
        "mse_after": result.metrics_after.mean_squared_error,
        "warnings": result.warnings,
    }


def _output_file_map(paths: RegistrationOutputPaths) -> dict[str, Path]:
    return {
        "registered_moving_volume": Path(paths.registered_moving_volume),
        "transform": Path(paths.transform),
        "metrics": Path(paths.metrics),
        "metadata": Path(paths.metadata),
        "report": Path(paths.report),
        "overlay_mid_axial": Path(paths.overlay_mid_axial),
        "difference_mid_axial": Path(paths.difference_mid_axial),
        "fixed_mid_axial": Path(paths.fixed_mid_axial),
        "moving_mid_axial": Path(paths.moving_mid_axial),
        "registered_mid_axial": Path(paths.registered_mid_axial),
    }


def _write_numpy_atomic(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array)
    tmp_path.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
