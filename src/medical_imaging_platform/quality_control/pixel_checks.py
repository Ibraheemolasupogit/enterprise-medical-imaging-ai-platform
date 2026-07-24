"""Pixel-array technical integrity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from medical_imaging_platform.ingestion.loader import load_dicom
from medical_imaging_platform.ingestion.models import DicomFileMetadata
from medical_imaging_platform.quality_control.models import QualityControlConfig, QualityFinding
from medical_imaging_platform.quality_control.rules import rule_info


def run_pixel_checks(
    metadata: list[DicomFileMetadata],
    config: QualityControlConfig,
    *,
    max_file_size_bytes: int,
) -> tuple[list[QualityFinding], dict[str, object]]:
    """Run full pixel-array integrity checks when requested."""
    findings: list[QualityFinding] = []
    readable = 0
    constant = 0
    min_value: float | None = None
    max_value: float | None = None
    for item in metadata:
        if not item.has_pixel_data:
            _append(
                findings,
                "DICOM-QC-PIX-001",
                "FAIL",
                "PixelData is missing.",
                [item.file_path],
                False,
                True,
                "Review source DICOM file.",
            )
            continue
        try:
            dataset = load_dicom(
                Path(item.file_path),
                header_only=False,
                max_file_size_bytes=max_file_size_bytes,
            )
            pixel_array = np.asarray(dataset.pixel_array)
            readable += 1
        except Exception as exc:
            _append(
                findings,
                "DICOM-QC-PIX-002",
                "FAIL",
                "Pixel array cannot be decoded.",
                [item.file_path],
                str(exc),
                "readable pixel array",
                "Review transfer syntax and pixel data.",
            )
            continue
        if pixel_array.ndim != 2:
            _append(
                findings,
                "DICOM-QC-PIX-003",
                "FAIL",
                "Pixel array dimensions are not expected for one CT slice.",
                [item.file_path],
                pixel_array.shape,
                "2D",
                "Review image encoding.",
            )
        if (
            item.rows is not None
            and item.columns is not None
            and pixel_array.shape
            != (
                item.rows,
                item.columns,
            )
        ):
            _append(
                findings,
                "DICOM-QC-PIX-003",
                "FAIL",
                "Pixel array shape does not match Rows and Columns.",
                [item.file_path],
                pixel_array.shape,
                (item.rows, item.columns),
                "Review pixel metadata consistency.",
            )
        if not np.all(np.isfinite(pixel_array)):
            _append(
                findings,
                "DICOM-QC-PIX-004",
                "FAIL",
                "Pixel values include non-finite values.",
                [item.file_path],
                "non-finite",
                "finite",
                "Regenerate or reject the fixture.",
            )
        current_min = float(np.min(pixel_array))
        current_max = float(np.max(pixel_array))
        min_value = current_min if min_value is None else min(min_value, current_min)
        max_value = current_max if max_value is None else max(max_value, current_max)
        if current_min == current_max:
            constant += 1
            _append(
                findings,
                "DICOM-QC-PIX-004",
                "FAIL",
                "Pixel slice is constant; this is a technical warning only.",
                [item.file_path],
                current_min,
                "non-constant engineering fixture",
                "Review fixture generation or source image.",
            )
        lower, upper = config.pixel_value_bounds
        if current_min < lower or current_max > upper:
            _append(
                findings,
                "DICOM-QC-PIX-004",
                "FAIL",
                "Pixel values are outside configured engineering bounds.",
                [item.file_path],
                [current_min, current_max],
                [lower, upper],
                "Review pixel data and configured technical bounds.",
            )
        if item.rescale_slope is None or item.rescale_intercept is None:
            _append(
                findings,
                "DICOM-QC-PIX-004",
                "FAIL",
                "RescaleSlope or RescaleIntercept is missing; HU conversion is not performed.",
                [item.file_path],
                None,
                "present or safely defaultable",
                "Review rescale metadata.",
            )
    return findings, {
        "readable_pixel_array_count": readable,
        "constant_slice_count": constant,
        "minimum_pixel_value": min_value,
        "maximum_pixel_value": max_value,
    }


def _append(
    findings: list[QualityFinding],
    rule_id: str,
    status: str,
    message: str,
    affected_files: list[str],
    observed_value: object,
    expected_value: object,
    remediation: str,
) -> None:
    category, severity, _ = rule_info(rule_id)
    findings.append(
        QualityFinding(
            rule_id=rule_id,
            category=category,
            severity=severity,
            status=status,  # type: ignore[arg-type]
            message=message,
            affected_files=affected_files,
            observed_value=observed_value,
            expected_value=expected_value,
            remediation=remediation,
        )
    )
