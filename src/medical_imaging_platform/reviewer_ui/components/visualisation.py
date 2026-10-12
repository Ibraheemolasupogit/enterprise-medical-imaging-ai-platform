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


def render_slice_with_boxes(
    st_module: Any, image: np.ndarray, boxes: list[list[float]], title: str
) -> None:
    """Render one 2D slice with half-open xyxy detection boxes."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if image.ndim != 2:
        raise ValueError("Detection overlays require a 2D image.")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray")
    for box in boxes:
        x1, y1, x2, y2 = box
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="lime",
                linewidth=1.5,
            )
        )
    ax.set_title(title)
    ax.axis("off")
    st_module.pyplot(fig)
    plt.close(fig)
