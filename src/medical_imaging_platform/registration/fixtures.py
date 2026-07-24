"""Synthetic preprocessing-compatible fixtures for registration validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medical_imaging_platform.preprocessing.models import (
    CropPaddingSummary,
    GeometrySummary,
    IntensityTransformSummary,
    PixelConversionSummary,
    PreprocessingOutputPaths,
    PreprocessingResult,
)
from medical_imaging_platform.synthetic.manifest import sha256_file


def generate_registration_fixture_pair(
    output_dir: Path, *, overwrite: bool = False
) -> tuple[Path, Path]:
    """Generate deterministic fixed/moving preprocessing outputs for registration."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Registration fixture directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed = _synthetic_volume()
    moving = np.roll(fixed, shift=(2, -1, 3), axis=(0, 1, 2))
    fixed_dir = output_dir / "fixed"
    moving_dir = output_dir / "moving"
    _write_preprocessed_fixture(fixed_dir, fixed, run_id="preprocess-registration-fixed")
    _write_preprocessed_fixture(moving_dir, moving, run_id="preprocess-registration-moving")
    return fixed_dir, moving_dir


def _synthetic_volume() -> np.ndarray:
    z, y, x = np.indices((24, 24, 24))
    sphere = ((z - 11) ** 2 + (y - 12) ** 2 + (x - 10) ** 2) <= 25
    smaller = ((z - 14) ** 2 + (y - 8) ** 2 + (x - 15) ** 2) <= 9
    volume = np.zeros((24, 24, 24), dtype=np.float32)
    volume[sphere] = 1.0
    volume[smaller] = 0.55
    return volume


def _write_preprocessed_fixture(output_dir: Path, volume: np.ndarray, *, run_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    volume_path = output_dir / "volume.npy"
    metadata_path = output_dir / "metadata.json"
    report_path = output_dir / "preprocessing_report.md"
    with volume_path.open("wb") as handle:
        np.save(handle, volume.astype(np.float32))
    report_path.write_text("Synthetic registration preprocessing fixture.\n", encoding="utf-8")
    result = _preprocessing_result(output_dir, run_id, volume)
    result = result.model_copy(
        update={
            "checksums": {
                "volume": sha256_file(volume_path),
                "report": sha256_file(report_path),
            }
        }
    )
    metadata_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _preprocessing_result(output_dir: Path, run_id: str, volume: np.ndarray) -> PreprocessingResult:
    shape = (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
    geometry = GeometrySummary(
        spacing_mm=(1.0, 1.0, 1.0),
        spacing_source="synthetic_registration_fixture",
        slice_spacing_median_mm=1.0,
        slice_spacing_min_mm=1.0,
        slice_spacing_max_mm=1.0,
        slice_spacing_irregular=False,
        slice_spacing_reliable=True,
        slice_thickness_fallback_used=False,
        row_direction_cosines=(1.0, 0.0, 0.0),
        column_direction_cosines=(0.0, 1.0, 0.0),
        slice_normal=(0.0, 0.0, 1.0),
        image_positions_patient=[(0.0, 0.0, float(index)) for index in range(shape[0])],
        orientation_classification="axial_like",
        geometry_note="Synthetic registration fixture; no anatomical claim.",
    )
    return PreprocessingResult(
        run_id=run_id,
        study_instance_uid=f"1.2.826.0.1.3680043.10.54321.6.{1 if 'fixed' in run_id else 2}",
        series_instance_uid=f"1.2.826.0.1.3680043.10.54321.6.{3 if 'fixed' in run_id else 4}",
        source_quality_report_id="synthetic-registration-qc",
        source_quality_status="PASS",
        volume_shape=shape,
        spacing_mm=(1.0, 1.0, 1.0),
        axis_order="z,y,x",
        orientation_classification="axial_like",
        geometry=geometry,
        pixel_conversion=PixelConversionSummary(
            output_dtype="float32",
            terminology="synthetic registration intensity",
            raw_range=(float(np.min(volume)), float(np.max(volume))),
            converted_range=(float(np.min(volume)), float(np.max(volume))),
            slope_values=[1.0],
            intercept_values=[0.0],
            defaulted_slope_sop_uids=[],
            defaulted_intercept_sop_uids=[],
        ),
        intensity_transform=IntensityTransformSummary(
            profile_name="none",
            clipping_lower=None,
            clipping_upper=None,
            normalisation_mode="none",
            input_range=(float(np.min(volume)), float(np.max(volume))),
            clipped_range=(float(np.min(volume)), float(np.max(volume))),
            output_range=(float(np.min(volume)), float(np.max(volume))),
            clipped_voxel_count=0,
            mean_before_normalisation=float(np.mean(volume)),
            std_before_normalisation=float(np.std(volume)),
            fallback_used=False,
            output_dtype="float32",
        ),
        crop_padding=CropPaddingSummary(
            crop_mode="none",
            input_shape=shape,
            crop_bounds_zyx={"z": (0, shape[0]), "y": (0, shape[1]), "x": (0, shape[2])},
            crop_offsets_zyx=(0, 0, 0),
            cropped_shape=shape,
            padding_value=0.0,
            pad_widths_zyx=((0, 0), (0, 0), (0, 0)),
            output_shape=shape,
        ),
        slice_count_input=shape[0],
        slice_count_output=shape[0],
        sop_instance_uid_to_index={},
        unreadable_files=[],
        excluded_files=[],
        output_paths=PreprocessingOutputPaths(
            output_dir=str(output_dir),
            volume=str(output_dir / "volume.npy"),
            metadata=str(output_dir / "metadata.json"),
            report=str(output_dir / "preprocessing_report.md"),
        ),
        checksums={},
        policy_version="synthetic-registration-preprocessing-v0.6.0",
        generated_at="2026-01-01T00:00:00+00:00",
        warnings=[],
        override_used=False,
        override_reason=None,
    )
