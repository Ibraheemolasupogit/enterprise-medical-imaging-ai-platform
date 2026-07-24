"""Deterministic NumPy review arrays for longitudinal analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_review_arrays(
    output_dir: Path,
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
) -> dict[str, str]:
    """Write deterministic engineering review arrays and return relative paths."""
    review_dir = output_dir / "review_arrays"
    review_dir.mkdir(parents=True, exist_ok=True)
    previous = previous_mask.astype(np.float32)
    current = current_mask.astype(np.float32)
    z_previous = previous.shape[0] // 2
    z_current = current.shape[0] // 2
    arrays = {
        "previous_mid_slice": previous[z_previous],
        "current_mid_slice": current[z_current],
        "previous_mask_overlay": previous[z_previous] * 1.0,
        "current_mask_overlay": current[z_current] * 1.0,
        "registered_difference": current - previous
        if current.shape == previous.shape
        else np.zeros_like(current),
    }
    paths: dict[str, str] = {}
    for name, array in arrays.items():
        path = review_dir / f"{name}.npy"
        tmp = path.with_name(f".{path.name}.tmp")
        with tmp.open("wb") as handle:
            np.save(handle, array.astype(np.float32))
        tmp.replace(path)
        paths[name] = str(path.relative_to(output_dir))
    return paths
