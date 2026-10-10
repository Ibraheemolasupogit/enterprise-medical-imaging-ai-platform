"""Output writing and preview helpers for preprocessing baseline evidence."""

from __future__ import annotations

import json
import os
import struct
import zlib
from pathlib import Path

import numpy as np

from medical_imaging_platform.preprocessing_baseline.models import (
    BaselineOutputPaths,
    PreprocessingBaselineResult,
)
from medical_imaging_platform.preprocessing_baseline.report import (
    render_preprocessing_baseline_report,
)
from medical_imaging_platform.synthetic.manifest import sha256_file
from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class PreprocessingBaselineOutputError(MedicalImagingPlatformError):
    """Raised when M19 baseline evidence cannot be written or validated."""


def build_output_paths(output_root: Path, baseline_run_id: str) -> BaselineOutputPaths:
    """Build protected output paths for one baseline run."""
    root = output_root.resolve()
    output_dir = (root / baseline_run_id).resolve()
    if os.path.commonpath([str(root), str(output_dir)]) != str(root):
        raise PreprocessingBaselineOutputError(
            "Baseline output path escapes configured output root."
        )
    evidence_dir = output_dir / "evidence"
    return BaselineOutputPaths(
        output_dir=str(output_dir),
        metrics=str(output_dir / "preprocessing_baseline_metrics.json"),
        summary=str(output_dir / "before_after_summary.json"),
        report=str(output_dir / "preprocessing_baseline_report.md"),
        checksum_manifest=str(evidence_dir / "checksum_manifest.json"),
    )


def write_preprocessing_baseline_outputs(
    result: PreprocessingBaselineResult,
    transformed_volumes: dict[str, np.ndarray],
    preview_arrays: dict[str, np.ndarray],
    *,
    overwrite: bool,
) -> PreprocessingBaselineResult:
    """Write deterministic baseline evidence and checksums."""
    output_dir = Path(result.output_paths.output_dir)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    target_paths = _output_file_map(result)
    if not overwrite and any(path.exists() for path in target_paths.values()):
        raise PreprocessingBaselineOutputError(
            f"Preprocessing baseline output already exists in {output_dir}"
        )

    for key, array in sorted(transformed_volumes.items()):
        _write_numpy_atomic(target_paths[key], array.astype(np.float32))
    for key, array in sorted(preview_arrays.items()):
        _write_png(target_paths[key], array)

    _write_text_atomic(
        Path(result.output_paths.metrics),
        json.dumps(_metrics_payload(result), indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        Path(result.output_paths.summary),
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    checksums = {
        key: sha256_file(path)
        for key, path in sorted(target_paths.items())
        if key not in {"summary", "report", "checksum_manifest"}
    }
    result = result.model_copy(update={"checksums": checksums})
    _write_text_atomic(
        Path(result.output_paths.report), render_preprocessing_baseline_report(result)
    )
    checksums = {**result.checksums, "report": sha256_file(Path(result.output_paths.report))}
    result = result.model_copy(update={"checksums": checksums})
    _write_text_atomic(
        Path(result.output_paths.summary),
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        Path(result.output_paths.checksum_manifest),
        json.dumps(result.checksums, indent=2, sort_keys=True) + "\n",
    )
    return result


def preview_slice(array: np.ndarray, index: int | None) -> np.ndarray:
    """Return a representative 2D slice for preview evidence."""
    volume = np.asarray(array, dtype=np.float32)
    if volume.ndim == 2:
        return volume
    selected = volume.shape[0] // 2 if index is None else max(0, min(index, volume.shape[0] - 1))
    return np.asarray(volume[selected], dtype=np.float32)


def _metrics_payload(result: PreprocessingBaselineResult) -> dict[str, object]:
    return {
        "baseline_run_id": result.baseline_run_id,
        "policy_version": result.policy_version,
        "sources": [
            {
                "source_preprocessing_run_id": source.source_preprocessing_run_id,
                "before_metrics": source.before_metrics.model_dump(mode="json"),
                "after_metrics": source.after_metrics.model_dump(mode="json"),
                "metric_deltas": source.metric_deltas,
            }
            for source in result.sources
        ],
    }


def _output_file_map(result: PreprocessingBaselineResult) -> dict[str, Path]:
    paths = {
        "metrics": Path(result.output_paths.metrics),
        "summary": Path(result.output_paths.summary),
        "report": Path(result.output_paths.report),
        "checksum_manifest": Path(result.output_paths.checksum_manifest),
    }
    for source in result.sources:
        source_key = source.source_preprocessing_run_id
        paths[f"{source_key}_transformed_volume"] = Path(source.transformed_volume)
        if source.preview_before is not None:
            paths[f"{source_key}_preview_before"] = Path(source.preview_before)
        if source.preview_after is not None:
            paths[f"{source_key}_preview_after"] = Path(source.preview_after)
    return paths


def _write_numpy_atomic(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array)
    tmp_path.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _write_png(path: Path, array: np.ndarray) -> None:
    image = _normalise_uint8(array)
    height, width = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw)),
            _png_chunk(b"IEND", b""),
        ]
    )
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_bytes(png)
    tmp_path.replace(path)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _normalise_uint8(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if minimum == maximum:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image - minimum) / (maximum - minimum), 0.0, 1.0)
    return np.asarray(np.rint(scaled * 255.0), dtype=np.uint8)
