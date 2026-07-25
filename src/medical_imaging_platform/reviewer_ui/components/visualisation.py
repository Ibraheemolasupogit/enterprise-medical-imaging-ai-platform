"""Matplotlib visualisation helpers for bounded arrays."""

from __future__ import annotations

from typing import Any

import numpy as np


def axial_slice(array: np.ndarray, index: int | None = None) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError("Only 3D arrays can be visualised.")
    selected = array.shape[0] // 2 if index is None else max(0, min(index, array.shape[0] - 1))
    return np.asarray(array[selected], dtype=np.float32)


def render_axial_slice(st_module: Any, array: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(axial_slice(array), cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    st_module.pyplot(fig)
    plt.close(fig)
