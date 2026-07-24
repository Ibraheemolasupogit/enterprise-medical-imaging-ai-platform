"""Deterministic preprocessing export and validation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from medical_imaging_platform.preprocessing.errors import PreprocessingOutputError
from medical_imaging_platform.preprocessing.models import (
    PreprocessingOutputPaths,
    PreprocessingResult,
)
from medical_imaging_platform.preprocessing.report import render_preprocessing_report
from medical_imaging_platform.synthetic.manifest import sha256_file


def build_output_paths(output_root: Path, run_id: str) -> PreprocessingOutputPaths:
    """Build output paths with path traversal protection."""
    root = output_root.resolve()
    output_dir = (root / run_id).resolve()
    if os.path.commonpath([str(root), str(output_dir)]) != str(root):
        raise PreprocessingOutputError("Preprocessing output path escapes configured output root.")
    return PreprocessingOutputPaths(
        output_dir=str(output_dir),
        volume=str(output_dir / "volume.npy"),
        metadata=str(output_dir / "metadata.json"),
        report=str(output_dir / "preprocessing_report.md"),
    )


def write_preprocessing_outputs(
    volume: np.ndarray,
    result: PreprocessingResult,
    *,
    overwrite: bool,
) -> PreprocessingResult:
    """Write NumPy volume, metadata JSON, and Markdown report atomically where practical."""
    output_dir = Path(result.output_paths.output_dir)
    volume_path = Path(result.output_paths.volume)
    metadata_path = Path(result.output_paths.metadata)
    report_path = Path(result.output_paths.report)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [volume_path, metadata_path, report_path]
    if not overwrite and any(path.exists() for path in targets):
        raise PreprocessingOutputError(f"Preprocessing output already exists in {output_dir}")

    _write_numpy_atomic(volume_path, volume)
    reloaded = np.load(volume_path)
    if tuple(reloaded.shape) != tuple(volume.shape):
        raise PreprocessingOutputError("Reloaded NumPy volume shape does not match exported shape.")

    checksums = {"volume": sha256_file(volume_path)}
    result = result.model_copy(update={"checksums": checksums})
    _write_text_atomic(
        metadata_path,
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(report_path, render_preprocessing_report(result))
    result = result.model_copy(
        update={
            "checksums": {
                **result.checksums,
                "report": sha256_file(report_path),
            }
        }
    )
    _write_text_atomic(
        metadata_path,
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    return result


def validate_preprocessed_volume(output_dir: Path) -> PreprocessingResult:
    """Validate exported preprocessing artefacts and checksums."""
    metadata_path = output_dir / "metadata.json"
    volume_path = output_dir / "volume.npy"
    report_path = output_dir / "preprocessing_report.md"
    if not metadata_path.exists() or not volume_path.exists() or not report_path.exists():
        raise PreprocessingOutputError("Preprocessed output is missing required artefacts.")
    try:
        result = PreprocessingResult.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        volume = np.load(volume_path)
    except Exception as exc:
        raise PreprocessingOutputError(f"Preprocessed output cannot be loaded: {exc}") from exc
    if tuple(volume.shape) != result.volume_shape:
        raise PreprocessingOutputError("Volume shape does not match preprocessing metadata.")
    for key, path in {
        "volume": volume_path,
        "report": report_path,
    }.items():
        expected = result.checksums.get(key)
        if expected is not None and sha256_file(path) != expected:
            raise PreprocessingOutputError(f"Checksum mismatch for {key}.")
    return result


def inspect_preprocessed_volume(output_dir: Path) -> dict[str, object]:
    """Return a compact deterministic inspection summary."""
    result = validate_preprocessed_volume(output_dir)
    volume = np.load(output_dir / "volume.npy")
    return {
        "run_id": result.run_id,
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "range": [float(np.min(volume)), float(np.max(volume))],
        "spacing_mm": list(result.spacing_mm),
        "axis_order": result.axis_order,
        "source_quality_status": result.source_quality_status,
        "warnings": result.warnings,
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
