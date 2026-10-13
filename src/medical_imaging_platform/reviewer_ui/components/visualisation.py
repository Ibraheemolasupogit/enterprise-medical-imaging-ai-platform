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


def render_segmentation_overlay(
    st_module: Any, source: np.ndarray, mask: np.ndarray, *, slice_index: int, opacity: float = 0.35
) -> None:
    """Display a source slice and binary prediction without changing either array."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    image = axial_slice(source, slice_index)
    if mask.shape != image.shape or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError("Mask preview must be binary and match the source slice.")
    if not 0 <= opacity <= 1:
        raise ValueError("Overlay opacity must be between zero and one.")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax in axes:
        ax.imshow(image, cmap="gray")
        ax.axis("off")
    axes[0].set_title(f"Source axial slice {slice_index}")
    axes[1].imshow(
        np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=opacity, vmin=0, vmax=1
    )
    axes[1].set_title("Predicted mask overlay — engineering review")
    st_module.pyplot(fig)
    plt.close(fig)
