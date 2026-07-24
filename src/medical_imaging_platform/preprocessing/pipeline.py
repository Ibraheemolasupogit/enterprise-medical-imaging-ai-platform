"""Milestone 5 preprocessing orchestration."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from medical_imaging_platform.preprocessing.assembly import assemble_selected_series
from medical_imaging_platform.preprocessing.cropping import apply_crop_and_padding
from medical_imaging_platform.preprocessing.export import (
    build_output_paths,
    write_preprocessing_outputs,
)
from medical_imaging_platform.preprocessing.intensity import apply_intensity_transform
from medical_imaging_platform.preprocessing.models import PreprocessingConfig, PreprocessingResult
from medical_imaging_platform.quality_control.models import QualityControlConfig


def preprocess_dicom_series(
    input_dir: Path,
    *,
    output_root: Path,
    preprocessing_config: PreprocessingConfig,
    qc_config: QualityControlConfig,
    dicom_max_files: int,
    max_file_size_bytes: int,
    study_uid: str | None = None,
    series_uid: str | None = None,
    intensity_profile: str | None = None,
    crop_mode: str | None = None,
    overwrite: bool | None = None,
    quality_override: bool = False,
    override_reason: str | None = None,
) -> PreprocessingResult:
    """Run the end-to-end deterministic preprocessing foundation."""
    effective_config = preprocessing_config
    if crop_mode is not None:
        effective_config = effective_config.model_copy(update={"crop_mode": crop_mode})
    assembled = assemble_selected_series(
        input_dir,
        study_uid=study_uid,
        series_uid=series_uid,
        preprocessing_config=effective_config,
        dicom_max_files=dicom_max_files,
        max_file_size_bytes=max_file_size_bytes,
        qc_config=qc_config,
        quality_override=quality_override,
    )
    transformed, intensity_summary, intensity_warnings = apply_intensity_transform(
        assembled.volume,
        config=effective_config,
        profile_name=intensity_profile,
    )
    cropped, crop_summary, crop_warnings = apply_crop_and_padding(
        transformed, config=effective_config
    )
    run_id = _run_id(
        assembled.study_instance_uid,
        assembled.series_instance_uid,
        effective_config.policy_version,
        _shape3(cropped.shape),
    )
    paths = build_output_paths(output_root, run_id)
    warnings = sorted(set(assembled.warnings + intensity_warnings + crop_warnings))
    result = PreprocessingResult(
        run_id=run_id,
        study_instance_uid=assembled.study_instance_uid,
        series_instance_uid=assembled.series_instance_uid,
        source_quality_report_id=assembled.source_quality_report_id,
        source_quality_status=assembled.source_quality_status,
        volume_shape=_shape3(cropped.shape),
        spacing_mm=assembled.geometry.spacing_mm,
        axis_order="z,y,x",
        orientation_classification=assembled.geometry.orientation_classification,
        geometry=assembled.geometry,
        pixel_conversion=assembled.pixel_conversion,
        intensity_transform=intensity_summary,
        crop_padding=crop_summary,
        slice_count_input=assembled.slice_count_input,
        slice_count_output=int(cropped.shape[0]),
        sop_instance_uid_to_index=assembled.sop_instance_uid_to_index,
        unreadable_files=assembled.unreadable_files,
        excluded_files=assembled.excluded_files,
        output_paths=paths,
        checksums={},
        policy_version=effective_config.policy_version,
        generated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        warnings=warnings,
        override_used=assembled.override_used,
        override_reason=override_reason if assembled.override_used else None,
    )
    return write_preprocessing_outputs(
        cropped,
        result,
        overwrite=effective_config.overwrite if overwrite is None else overwrite,
    )


def _run_id(
    study_uid: str, series_uid: str, policy_version: str, shape: tuple[int, int, int]
) -> str:
    digest = hashlib.sha256(
        "|".join(
            [study_uid, series_uid, policy_version, ",".join(str(item) for item in shape)]
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"preprocess-{digest}"


def _shape3(shape: tuple[int, ...]) -> tuple[int, int, int]:
    return (int(shape[0]), int(shape[1]), int(shape[2]))
