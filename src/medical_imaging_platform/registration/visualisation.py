"""Lightweight NumPy visual review artefacts."""

from __future__ import annotations

import numpy as np


def mid_axial_slices(
    fixed: np.ndarray, moving: np.ndarray, registered: np.ndarray
) -> dict[str, np.ndarray]:
    """Create deterministic mid-axial review arrays."""
    z = fixed.shape[0] // 2
    fixed_mid = fixed[z].astype(np.float32)
    moving_mid = moving[min(z, moving.shape[0] - 1)].astype(np.float32)
    registered_mid = registered[z].astype(np.float32)
    overlay = (0.5 * _scale01(fixed_mid) + 0.5 * _scale01(registered_mid)).astype(np.float32)
    difference = np.abs(fixed_mid - registered_mid).astype(np.float32)
    return {
        "fixed_mid_axial": fixed_mid,
        "moving_mid_axial": moving_mid,
        "registered_mid_axial": registered_mid,
        "overlay_mid_axial": overlay,
        "difference_mid_axial": difference,
    }


def _scale01(array: np.ndarray) -> np.ndarray:
    lower = float(np.min(array))
    upper = float(np.max(array))
    if upper == lower:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - lower) / (upper - lower)).astype(np.float32)
