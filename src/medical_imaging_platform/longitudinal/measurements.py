"""Spacing-aware synthetic lesion measurements."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from scipy import ndimage

from medical_imaging_platform.longitudinal.models import LesionMeasurement


class LongitudinalMeasurementError(ValueError):
    """Raised when lesion measurements cannot be computed."""


def validate_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a bool 3D mask after validating shape, finiteness, and binary values."""
    array = np.asarray(mask)
    if array.ndim != 3:
        raise LongitudinalMeasurementError("Mask must be 3D.")
    if not np.all(np.isfinite(array)):
        raise LongitudinalMeasurementError("Mask contains non-finite values.")
    unique = set(np.unique(array).tolist())
    if not unique <= {0, 1}:
        raise LongitudinalMeasurementError("Mask must be binary.")
    return array.astype(bool)


def validate_spacing(spacing_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(spacing_mm) != 3 or any(axis <= 0 or not np.isfinite(axis) for axis in spacing_mm):
        raise LongitudinalMeasurementError("Spacing must contain three finite positive values.")
    return cast(tuple[float, float, float], tuple(float(axis) for axis in spacing_mm))


def extract_components(mask: np.ndarray, minimum_component_voxels: int) -> list[np.ndarray]:
    """Extract deterministic connected components filtered by voxel count."""
    binary = validate_binary_mask(mask)
    labelled, count = ndimage.label(binary)
    components: list[np.ndarray] = []
    for index in range(1, int(count) + 1):
        component = labelled == index
        if int(np.count_nonzero(component)) >= minimum_component_voxels:
            components.append(component)
    return sorted(components, key=_component_sort_key)


def measure_components(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    timepoint: Literal["previous", "current"],
    minimum_component_voxels: int,
) -> list[LesionMeasurement]:
    """Measure each connected component, or one empty-mask record when no lesion exists."""
    spacing = validate_spacing(spacing_mm)
    components = extract_components(mask, minimum_component_voxels)
    if not components:
        return [_empty_measurement(timepoint)]
    component_count = len(components)
    return [
        measure_mask(
            component,
            spacing,
            lesion_id=f"{timepoint}-lesion-{index:03d}",
            timepoint=timepoint,
            component_index=index,
            connected_component_count=component_count,
        )
        for index, component in enumerate(components, start=1)
    ]


def measure_mask(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    lesion_id: str,
    timepoint: Literal["previous", "current"],
    component_index: int,
    connected_component_count: int,
) -> LesionMeasurement:
    """Compute deterministic physical measurements for one binary component."""
    binary = validate_binary_mask(mask)
    spacing = validate_spacing(spacing_mm)
    coords = np.argwhere(binary)
    voxel_count = int(coords.shape[0])
    if voxel_count == 0:
        return _empty_measurement(timepoint)
    voxel_volume = float(np.prod(np.asarray(spacing, dtype=float)))
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    dimensions = cast(
        tuple[float, float, float],
        tuple(float((maxs[axis] - mins[axis] + 1) * spacing[axis]) for axis in range(3)),
    )
    centroid = coords.mean(axis=0)
    centroid_tuple = cast(tuple[float, float, float], tuple(float(value) for value in centroid))
    centroid_physical = cast(
        tuple[float, float, float],
        tuple(float(centroid[axis] * spacing[axis]) for axis in range(3)),
    )
    return LesionMeasurement(
        lesion_id=lesion_id,
        timepoint=timepoint,
        component_index=component_index,
        voxel_count=voxel_count,
        physical_volume_mm3=float(voxel_count * voxel_volume),
        physical_volume_ml=float(voxel_count * voxel_volume / 1000.0),
        bounding_box_voxel=[[int(mins[axis]), int(maxs[axis] + 1)] for axis in range(3)],
        bounding_box_dimensions_mm=dimensions,
        maximum_3d_diameter_mm=maximum_3d_diameter(coords, spacing),
        axial_maximum_diameter_mm=axial_maximum_diameter(coords, spacing),
        centroid_voxel=centroid_tuple,
        centroid_physical_mm=centroid_physical,
        connected_component_count=connected_component_count,
        empty_mask=False,
    )


def maximum_3d_diameter(coords: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    """Return exact maximum pairwise 3D diameter for small synthetic components."""
    if len(coords) < 2:
        return 0.0
    physical = coords.astype(float) * np.asarray(spacing_mm, dtype=float)
    delta = physical[:, None, :] - physical[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    return float(np.max(distances))


def axial_maximum_diameter(coords: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    """Return maximum in-slice y/x diameter across axial z slices."""
    if len(coords) < 2:
        return 0.0
    best = 0.0
    yx_spacing = np.asarray(spacing_mm[1:], dtype=float)
    for z_value in sorted(set(coords[:, 0].tolist())):
        yx = coords[coords[:, 0] == z_value][:, 1:].astype(float) * yx_spacing
        if len(yx) < 2:
            continue
        delta = yx[:, None, :] - yx[None, :, :]
        best = max(best, float(np.max(np.sqrt(np.sum(delta * delta, axis=2)))))
    return best


def _empty_measurement(timepoint: Literal["previous", "current"]) -> LesionMeasurement:
    return LesionMeasurement(
        lesion_id=f"{timepoint}-lesion-empty",
        timepoint=timepoint,
        component_index=0,
        voxel_count=0,
        physical_volume_mm3=0.0,
        physical_volume_ml=0.0,
        bounding_box_voxel=[[0, 0], [0, 0], [0, 0]],
        bounding_box_dimensions_mm=(0.0, 0.0, 0.0),
        maximum_3d_diameter_mm=0.0,
        axial_maximum_diameter_mm=0.0,
        centroid_voxel=None,
        centroid_physical_mm=None,
        connected_component_count=0,
        empty_mask=True,
    )


def _component_sort_key(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    return (int(mins[0]), int(mins[1]), int(mins[2]), int(coords.shape[0]))
