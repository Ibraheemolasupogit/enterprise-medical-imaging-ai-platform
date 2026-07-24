"""ROI extraction and overlay helpers."""

from __future__ import annotations

import numpy as np

from medical_imaging_platform.localisation.models import (
    BoundingBox,
    LocalisationConfig,
    RoiExtraction,
)


def roi_size_voxels(
    spacing_mm_zyx: tuple[float, float, float], config: LocalisationConfig
) -> tuple[int, int, int]:
    """Resolve configured voxel or physical ROI size to voxel dimensions."""
    if config.roi_size_voxels is not None:
        return config.roi_size_voxels
    assert config.roi_size_mm is not None
    return (
        max(1, int(round(config.roi_size_mm[0] / spacing_mm_zyx[0]))),
        max(1, int(round(config.roi_size_mm[1] / spacing_mm_zyx[1]))),
        max(1, int(round(config.roi_size_mm[2] / spacing_mm_zyx[2]))),
    )


def bounding_box_for_centre(
    centre: tuple[int, int, int],
    size: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> tuple[BoundingBox, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """Return clipped half-open box and required padding widths."""
    bounds: list[tuple[int, int]] = []
    pads: list[tuple[int, int]] = []
    for axis in range(3):
        start = centre[axis] - size[axis] // 2
        end = start + size[axis]
        pad_before = max(0, -start)
        pad_after = max(0, end - shape[axis])
        bounds.append((max(0, start), min(shape[axis], end)))
        pads.append((pad_before, pad_after))
    return (
        BoundingBox(z=bounds[0], y=bounds[1], x=bounds[2]),
        (pads[0], pads[1], pads[2]),
    )


def extract_roi(
    volume: np.ndarray,
    *,
    centre: tuple[int, int, int],
    spacing_mm_zyx: tuple[float, float, float],
    config: LocalisationConfig,
) -> tuple[np.ndarray, RoiExtraction]:
    """Extract deterministic ROI with padding near boundaries."""
    size = roi_size_voxels(spacing_mm_zyx, config)
    shape = (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
    box, pads = bounding_box_for_centre(centre, size, shape)
    cropped = volume[box.z[0] : box.z[1], box.y[0] : box.y[1], box.x[0] : box.x[1]]
    if cropped.size == 0:
        raise ValueError("ROI extraction produced an empty output.")
    roi = np.pad(
        cropped,
        pads,
        mode="constant",
        constant_values=float(config.padding_value),
    )
    padding_voxels = int(roi.size - cropped.size)
    extraction = RoiExtraction(
        roi_shape=(int(roi.shape[0]), int(roi.shape[1]), int(roi.shape[2])),
        crop_bounds_zyx=box,
        crop_offsets_zyx=(box.z[0], box.y[0], box.x[0]),
        pad_widths_zyx=pads,
        padding_fraction=float(padding_voxels / roi.size),
        source_spacing_mm_zyx=spacing_mm_zyx,
    )
    return np.asarray(roi, dtype=volume.dtype), extraction


def bounding_box_mm(
    box: BoundingBox, spacing_mm_zyx: tuple[float, float, float]
) -> dict[str, tuple[float, float]]:
    return {
        "z": (box.z[0] * spacing_mm_zyx[0], box.z[1] * spacing_mm_zyx[0]),
        "y": (box.y[0] * spacing_mm_zyx[1], box.y[1] * spacing_mm_zyx[1]),
        "x": (box.x[0] * spacing_mm_zyx[2], box.x[1] * spacing_mm_zyx[2]),
    }


def overlay_mid_slice(
    volume: np.ndarray, box: BoundingBox, centre: tuple[int, int, int]
) -> np.ndarray:
    """Create a simple mid-slice overlay array with box outline and centre marker."""
    z_index = centre[0]
    base = _scale01(volume[z_index]).astype(np.float32)
    overlay = base.copy()
    y0, y1 = box.y
    x0, x1 = box.x
    overlay[max(y0, 0) : min(y1, overlay.shape[0]), max(x0, 0)] = 1.0
    overlay[max(y0, 0) : min(y1, overlay.shape[0]), min(x1 - 1, overlay.shape[1] - 1)] = 1.0
    overlay[max(y0, 0), max(x0, 0) : min(x1, overlay.shape[1])] = 1.0
    overlay[min(y1 - 1, overlay.shape[0] - 1), max(x0, 0) : min(x1, overlay.shape[1])] = 1.0
    overlay[centre[1], centre[2]] = 1.0
    return overlay.astype(np.float32)


def _scale01(array: np.ndarray) -> np.ndarray:
    lower = float(np.min(array))
    upper = float(np.max(array))
    if lower == upper:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - lower) / (upper - lower)).astype(np.float32)
