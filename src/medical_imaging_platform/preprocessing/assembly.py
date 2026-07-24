"""DICOM series selection, quality gating, and volume assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medical_imaging_platform.ingestion.discovery import discover_dicom_series
from medical_imaging_platform.ingestion.loader import load_dicom
from medical_imaging_platform.ingestion.metadata import extract_metadata
from medical_imaging_platform.ingestion.models import DicomFileMetadata, DicomSeries
from medical_imaging_platform.ingestion.ordering import order_slices
from medical_imaging_platform.preprocessing.errors import (
    PreprocessingError,
    PreprocessingQualityError,
    PreprocessingRejectedError,
)
from medical_imaging_platform.preprocessing.geometry import summarise_geometry
from medical_imaging_platform.preprocessing.models import (
    GeometrySummary,
    PixelConversionSummary,
    PreprocessingConfig,
)
from medical_imaging_platform.quality_control.models import QualityControlConfig, QualityReport
from medical_imaging_platform.quality_control.pipeline import run_quality_control


@dataclass(frozen=True)
class AssembledVolume:
    """Assembled technical CT-like volume and source trace."""

    volume: np.ndarray
    study_instance_uid: str
    series_instance_uid: str
    geometry: GeometrySummary
    pixel_conversion: PixelConversionSummary
    sop_instance_uid_to_index: dict[str, int]
    source_quality_report_id: str | None
    source_quality_status: str | None
    slice_count_input: int
    unreadable_files: list[str]
    excluded_files: list[str]
    warnings: list[str]
    override_used: bool


def assemble_selected_series(
    input_dir: Path,
    *,
    study_uid: str | None,
    series_uid: str | None,
    preprocessing_config: PreprocessingConfig,
    dicom_max_files: int,
    max_file_size_bytes: int,
    qc_config: QualityControlConfig,
    quality_override: bool = False,
) -> AssembledVolume:
    """Assemble one selected DICOM series into a [z, y, x] NumPy volume."""
    discovery = discover_dicom_series(
        input_dir,
        max_files=dicom_max_files,
        max_file_size_bytes=max_file_size_bytes,
    )
    selected = _select_series(discovery.series, study_uid=study_uid, series_uid=series_uid)
    reports = run_quality_control(
        input_dir,
        output_dir=None,
        qc_config=qc_config,
        max_files=dicom_max_files,
        max_file_size_bytes=max_file_size_bytes,
        full_pixel_validation=True,
        overwrite=False,
    )
    quality_report = _matching_quality_report(reports, selected)
    override_used = _enforce_quality_gate(
        quality_report,
        preprocessing_config=preprocessing_config,
        requested_override=quality_override,
    )

    metadata, unreadable_files = _load_metadata(selected.files, max_file_size_bytes)
    _validate_metadata_consistency(metadata, selected)
    ordered = order_slices(metadata)
    ordering_errors = [issue.message for issue in ordered.issues if issue.severity == "ERROR"]
    warnings = [issue.message for issue in ordered.issues if issue.severity == "WARNING"]
    if ordering_errors and not override_used:
        raise PreprocessingError(
            "Ambiguous slice ordering; use the explicit engineering quality override only "
            f"after review: {ordering_errors}"
        )
    if ordering_errors:
        warnings.extend(f"Ordering override accepted: {message}" for message in ordering_errors)

    geometry, geometry_warnings = summarise_geometry(
        ordered.files,
        irregularity_tolerance_mm=preprocessing_config.spacing_irregularity_tolerance_mm,
    )
    warnings.extend(geometry_warnings)
    volume, conversion = _stack_pixels(
        ordered.files,
        max_file_size_bytes=max_file_size_bytes,
        preprocessing_config=preprocessing_config,
    )
    sop_to_index = {
        str(item.sop_instance_uid): index
        for index, item in enumerate(ordered.files)
        if item.sop_instance_uid is not None
    }
    return AssembledVolume(
        volume=volume,
        study_instance_uid=selected.study_instance_uid,
        series_instance_uid=selected.series_instance_uid,
        geometry=geometry,
        pixel_conversion=conversion,
        sop_instance_uid_to_index=sop_to_index,
        source_quality_report_id=quality_report.quality_report_id if quality_report else None,
        source_quality_status=quality_report.status if quality_report else None,
        slice_count_input=len(selected.files),
        unreadable_files=unreadable_files,
        excluded_files=[],
        warnings=warnings,
        override_used=override_used,
    )


def _select_series(
    series: list[DicomSeries], *, study_uid: str | None, series_uid: str | None
) -> DicomSeries:
    if not series:
        raise PreprocessingError("No DICOM series found in input directory.")
    if len(series) > 1 and (not study_uid or not series_uid):
        raise PreprocessingError(
            "Multiple DICOM series found; explicit --study-uid and --series-uid are required."
        )
    if study_uid == "auto" or series_uid == "auto":
        if len(series) != 1:
            raise PreprocessingError("Auto series selection is only allowed for a single series.")
        return series[0]
    if study_uid is None and series_uid is None:
        return series[0]
    matches = [
        item
        for item in series
        if item.study_instance_uid == study_uid and item.series_instance_uid == series_uid
    ]
    if len(matches) != 1:
        raise PreprocessingError("Explicit study/series UID did not match exactly one series.")
    return matches[0]


def _matching_quality_report(
    reports: list[QualityReport], selected: DicomSeries
) -> QualityReport | None:
    for report in reports:
        if (
            report.study_instance_uid == selected.study_instance_uid
            and report.series_instance_uid == selected.series_instance_uid
        ):
            return report
    return None


def _enforce_quality_gate(
    quality_report: QualityReport | None,
    *,
    preprocessing_config: PreprocessingConfig,
    requested_override: bool,
) -> bool:
    if quality_report is None:
        if preprocessing_config.require_quality_report:
            raise PreprocessingQualityError("No source quality report was available.")
        return False
    critical_failures = [
        finding.rule_id
        for finding in quality_report.findings
        if finding.severity == "CRITICAL" and finding.status == "FAIL"
    ]
    if quality_report.status == "REJECTED" or critical_failures:
        raise PreprocessingRejectedError(
            "Quality report rejected the series or found critical failures: "
            f"{critical_failures or quality_report.status}"
        )
    if quality_report.status == "FAIL":
        if requested_override and preprocessing_config.allow_quality_override:
            return True
        raise PreprocessingQualityError(
            "Quality report status FAIL blocks preprocessing without explicit override."
        )
    return False


def _load_metadata(
    files: list[str], max_file_size_bytes: int
) -> tuple[list[DicomFileMetadata], list[str]]:
    metadata: list[DicomFileMetadata] = []
    unreadable: list[str] = []
    for file_path in files:
        try:
            dataset = load_dicom(
                Path(file_path), header_only=False, max_file_size_bytes=max_file_size_bytes
            )
            metadata.append(extract_metadata(dataset, Path(file_path)))
        except Exception:
            unreadable.append(file_path)
    if unreadable:
        raise PreprocessingError(f"Unreadable DICOM files prevent assembly: {unreadable}")
    if not metadata:
        raise PreprocessingError("Selected series contains no readable slices.")
    return metadata, unreadable


def _validate_metadata_consistency(
    metadata: list[DicomFileMetadata], selected: DicomSeries
) -> None:
    rows = {item.rows for item in metadata}
    columns = {item.columns for item in metadata}
    pixel_representation = {item.pixel_representation for item in metadata}
    transfer_syntaxes = {item.transfer_syntax_uid for item in metadata}
    studies = {item.study_instance_uid for item in metadata}
    series = {item.series_instance_uid for item in metadata}
    if studies != {selected.study_instance_uid} or series != {selected.series_instance_uid}:
        raise PreprocessingError("Selected files contain mixed study or series UIDs.")
    if len(rows) != 1 or len(columns) != 1 or None in rows or None in columns:
        raise PreprocessingError("All slices must have consistent Rows and Columns.")
    if len(pixel_representation) != 1:
        raise PreprocessingError("All slices must have compatible PixelRepresentation metadata.")
    if len(transfer_syntaxes) != 1 or None in transfer_syntaxes:
        raise PreprocessingError("All slices must have a compatible TransferSyntaxUID.")
    if any(not item.has_pixel_data for item in metadata):
        raise PreprocessingError("All selected slices must contain PixelData.")


def _stack_pixels(
    ordered_metadata: list[DicomFileMetadata],
    *,
    max_file_size_bytes: int,
    preprocessing_config: PreprocessingConfig,
) -> tuple[np.ndarray, PixelConversionSummary]:
    slices: list[np.ndarray] = []
    raw_min = float("inf")
    raw_max = float("-inf")
    converted_min = float("inf")
    converted_max = float("-inf")
    slopes: list[float] = []
    intercepts: list[float] = []
    defaulted_slopes: list[str] = []
    defaulted_intercepts: list[str] = []

    for item in ordered_metadata:
        dataset = load_dicom(
            Path(item.file_path), header_only=False, max_file_size_bytes=max_file_size_bytes
        )
        raw = np.asarray(dataset.pixel_array)
        slope = item.rescale_slope
        intercept = item.rescale_intercept
        if slope is None:
            slope = preprocessing_config.missing_rescale_slope_default
            defaulted_slopes.append(str(item.sop_instance_uid))
        if intercept is None:
            intercept = preprocessing_config.missing_rescale_intercept_default
            defaulted_intercepts.append(str(item.sop_instance_uid))
        if not np.isfinite(slope) or not np.isfinite(intercept):
            raise PreprocessingError("Invalid numeric rescale slope/intercept metadata.")
        converted = raw.astype(preprocessing_config.pixel_dtype) * float(slope) + float(intercept)
        raw_min = min(raw_min, float(np.min(raw)))
        raw_max = max(raw_max, float(np.max(raw)))
        converted_min = min(converted_min, float(np.min(converted)))
        converted_max = max(converted_max, float(np.max(converted)))
        slopes.append(float(slope))
        intercepts.append(float(intercept))
        slices.append(converted.astype(preprocessing_config.pixel_dtype))

    volume = np.stack(slices, axis=0).astype(preprocessing_config.pixel_dtype)
    summary = PixelConversionSummary(
        output_dtype=str(volume.dtype),
        terminology="CT-like rescaled intensity",
        raw_range=(raw_min, raw_max),
        converted_range=(converted_min, converted_max),
        slope_values=slopes,
        intercept_values=intercepts,
        defaulted_slope_sop_uids=defaulted_slopes,
        defaulted_intercept_sop_uids=defaulted_intercepts,
    )
    return volume, summary
