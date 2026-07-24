"""Configurable binary mask post-processing."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from medical_imaging_platform.segmentation.models import SegmentationConfig


def postprocess_probability_map(
    probability: np.ndarray,
    *,
    config: SegmentationConfig,
    threshold: float | None = None,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Threshold and optionally clean a probability map."""
    cutoff = config.threshold if threshold is None else threshold
    mask = np.asarray(probability >= cutoff, dtype=bool)
    before = int(np.count_nonzero(mask))
    operations = [f"threshold>={cutoff:.4f}"]
    if config.minimum_component_voxels > 0 or config.keep_largest_component:
        mask, removed = _components(mask, config)
        operations.append(f"connected_components_removed={removed}")
    if config.fill_holes and np.any(mask):
        mask = ndimage.binary_fill_holes(mask)
        operations.append("fill_holes")
    after = int(np.count_nonzero(mask))
    warnings: list[str] = []
    if before > 0 and after == 0:
        warnings.append("Post-processing produced an empty mask.")
    return mask.astype(np.uint8), warnings, {"voxels_before": before, "voxels_after": after}


def _components(mask: np.ndarray, config: SegmentationConfig) -> tuple[np.ndarray, int]:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = np.ones(count + 1, dtype=bool)
    keep[0] = False
    if config.keep_largest_component:
        largest = int(np.argmax(sizes))
        keep[:] = False
        keep[largest] = True
    if config.minimum_component_voxels > 0:
        keep &= sizes >= config.minimum_component_voxels
    cleaned = keep[labels]
    removed = int(np.count_nonzero(mask) - np.count_nonzero(cleaned))
    return np.asarray(cleaned, dtype=bool), removed
