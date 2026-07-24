"""Geometry helpers for canonical [z, y, x] CT volume metadata."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from medical_imaging_platform.ingestion.models import DicomFileMetadata
from medical_imaging_platform.preprocessing.models import GeometrySummary

OrientationClassification = Literal["axial_like", "oblique", "indeterminate"]


def summarise_geometry(
    metadata: list[DicomFileMetadata], *, irregularity_tolerance_mm: float
) -> tuple[GeometrySummary, list[str]]:
    """Summarise source geometry without resampling or anatomical reorientation."""
    warnings: list[str] = []
    first = metadata[0]
    y_spacing, x_spacing = _pixel_spacing(first)
    row_cosines, column_cosines, normal = _directions(first)
    orientation_class = _classify_orientation(normal)

    positions = [item.image_position_patient for item in metadata]
    z_spacing, source, median, min_gap, max_gap, irregular, reliable, fallback = _z_spacing(
        metadata,
        normal,
        irregularity_tolerance_mm=irregularity_tolerance_mm,
    )
    if fallback:
        warnings.append("Slice spacing used SliceThickness fallback; no resampling was performed.")
    if irregular:
        warnings.append("Slice spacing appears irregular; spacing metadata is preserved only.")
    if orientation_class == "indeterminate":
        warnings.append("Orientation classification is indeterminate.")

    return (
        GeometrySummary(
            spacing_mm=(float(z_spacing), float(y_spacing), float(x_spacing)),
            spacing_source=source,
            slice_spacing_median_mm=median,
            slice_spacing_min_mm=min_gap,
            slice_spacing_max_mm=max_gap,
            slice_spacing_irregular=irregular,
            slice_spacing_reliable=reliable,
            slice_thickness_fallback_used=fallback,
            row_direction_cosines=row_cosines,
            column_direction_cosines=column_cosines,
            slice_normal=normal,
            image_positions_patient=positions,
            orientation_classification=orientation_class,
            geometry_note=(
                "Internal array axis order is [z, y, x]. Geometry is preserved for future "
                "registration/NIfTI work; this milestone does not perform full anatomical "
                "reorientation or spatial resampling."
            ),
        ),
        warnings,
    )


def _pixel_spacing(metadata: DicomFileMetadata) -> tuple[float, float]:
    if metadata.pixel_spacing is None:
        return (1.0, 1.0)
    return float(metadata.pixel_spacing[0]), float(metadata.pixel_spacing[1])


def _directions(
    metadata: DicomFileMetadata,
) -> tuple[
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
]:
    orientation = metadata.image_orientation_patient
    if orientation is None:
        return None, None, None
    row = np.array(orientation[:3], dtype=float)
    column = np.array(orientation[3:], dtype=float)
    normal_array = np.cross(row, column)
    norm = np.linalg.norm(normal_array)
    if math.isclose(norm, 0.0):
        return _triple(row), _triple(column), None
    normal = normal_array / norm
    return _triple(row), _triple(column), _triple(normal)


def _classify_orientation(
    normal: tuple[float, float, float] | None,
) -> OrientationClassification:
    if normal is None:
        return "indeterminate"
    abs_normal = tuple(abs(item) for item in normal)
    if abs_normal[2] >= 0.9:
        return "axial_like"
    return "oblique"


def _z_spacing(
    metadata: list[DicomFileMetadata],
    normal: tuple[float, float, float] | None,
    *,
    irregularity_tolerance_mm: float,
) -> tuple[float, str, float | None, float | None, float | None, bool, bool, bool]:
    if normal is not None and all(item.image_position_patient is not None for item in metadata):
        normal_array = np.array(normal, dtype=float)
        projections = np.array(
            [
                float(np.dot(np.array(item.image_position_patient, dtype=float), normal_array))
                for item in metadata
                if item.image_position_patient is not None
            ],
            dtype=float,
        )
        if len(projections) > 1:
            gaps = np.abs(np.diff(projections))
            median = float(np.median(gaps))
            min_gap = float(np.min(gaps))
            max_gap = float(np.max(gaps))
            irregular = (max_gap - min_gap) > irregularity_tolerance_mm
            reliable = median > 0.0 and not irregular
            return (
                median if median > 0.0 else _slice_thickness(metadata),
                "image_position_patient_projected_on_slice_normal",
                median,
                min_gap,
                max_gap,
                irregular,
                reliable,
                False,
            )
    thickness = _slice_thickness(metadata)
    return (thickness, "slice_thickness_fallback", None, None, None, False, False, True)


def _slice_thickness(metadata: list[DicomFileMetadata]) -> float:
    for item in metadata:
        if item.slice_thickness is not None:
            return float(item.slice_thickness)
    return 1.0


def _triple(array: np.ndarray) -> tuple[float, float, float]:
    return (float(array[0]), float(array[1]), float(array[2]))
