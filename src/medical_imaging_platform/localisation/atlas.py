"""Deterministic atlas-style adrenal-region placeholder localisation."""

from __future__ import annotations

import numpy as np

from medical_imaging_platform.localisation.models import LocalisationConfig, Side


def predict_centre_voxel(
    side: Side,
    *,
    volume_shape: tuple[int, int, int],
    spacing_mm_zyx: tuple[float, float, float],
    config: LocalisationConfig,
) -> tuple[int, int, int]:
    """Predict an atlas centre in [z, y, x] voxel coordinates."""
    relative = config.left_relative_centre if side == "left" else config.right_relative_centre
    physical_offset = (
        config.left_physical_offset_mm if side == "left" else config.right_physical_offset_mm
    )
    centre = []
    for axis, size in enumerate(volume_shape):
        base = relative[axis] * float(size - 1)
        offset_voxels = (
            physical_offset[axis] / spacing_mm_zyx[axis] if spacing_mm_zyx[axis] > 0 else 0.0
        )
        value = int(round(base + offset_voxels))
        centre.append(min(max(value, 0), size - 1))
    return (centre[0], centre[1], centre[2])


def centre_mm(
    centre_voxel: tuple[int, int, int], spacing_mm_zyx: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        float(centre_voxel[0] * spacing_mm_zyx[0]),
        float(centre_voxel[1] * spacing_mm_zyx[1]),
        float(centre_voxel[2] * spacing_mm_zyx[2]),
    )


def left_right_distance_mm(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    spacing_mm_zyx: tuple[float, float, float],
) -> float:
    left_mm = np.array(centre_mm(left, spacing_mm_zyx))
    right_mm = np.array(centre_mm(right, spacing_mm_zyx))
    return float(np.linalg.norm(left_mm - right_mm))
