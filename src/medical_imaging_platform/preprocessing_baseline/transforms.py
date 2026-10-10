"""Classical deterministic transforms for copies of preprocessed volumes."""

from __future__ import annotations

from typing import cast

import numpy as np

from medical_imaging_platform.preprocessing_baseline.metrics import validate_numeric_array
from medical_imaging_platform.preprocessing_baseline.models import (
    BaselineTransformName,
    PreprocessingBaselineConfig,
    TransformRecord,
)


def apply_baseline_transforms(
    volume: np.ndarray, config: PreprocessingBaselineConfig
) -> tuple[np.ndarray, list[TransformRecord], list[str]]:
    """Apply configured classical transforms to a copy of an existing preprocessed volume."""
    working = validate_numeric_array(volume)
    transform_chain: list[TransformRecord] = [
        TransformRecord(name="noop", parameters={"reason": "source copy created"})
    ]
    warnings: list[str] = []
    source_min = float(np.min(working))
    source_max = float(np.max(working))

    if config.denoise.enabled:
        if config.denoise.method == "median":
            working = _median_filter(working, config.denoise.kernel_size)
            name: BaselineTransformName = "median_denoise"
        else:
            working = _mean_filter(working, config.denoise.kernel_size)
            name = "mean_denoise"
        transform_chain.append(
            TransformRecord(
                name=name,
                parameters={
                    "kernel_size": config.denoise.kernel_size,
                    "padding": "edge",
                    "applied_to": "intensity volume copy",
                },
            )
        )

    if config.enhancement.enabled:
        if config.enhancement.method == "histogram_equalization":
            working = _histogram_equalize(working, config.enhancement.histogram_bins)
            transform_name: BaselineTransformName = "histogram_equalization"
            parameters: dict[str, object] = {"histogram_bins": config.enhancement.histogram_bins}
        elif config.enhancement.method == "clahe":
            working = _clahe_like(
                working,
                tiles=config.enhancement.clahe_tiles,
                bins=config.enhancement.histogram_bins,
                clip_limit=config.enhancement.clahe_clip_limit,
            )
            transform_name = "clahe"
            parameters = {
                "histogram_bins": config.enhancement.histogram_bins,
                "clahe_tiles": config.enhancement.clahe_tiles,
                "clahe_clip_limit": config.enhancement.clahe_clip_limit,
                "scope": "per axial slice tile",
            }
        else:
            working = _percentile_stretch(
                working,
                config.enhancement.lower_percentile,
                config.enhancement.upper_percentile,
            )
            transform_name = "percentile_stretch"
            parameters = {
                "lower_percentile": config.enhancement.lower_percentile,
                "upper_percentile": config.enhancement.upper_percentile,
            }
        transform_chain.append(TransformRecord(name=transform_name, parameters=parameters))

    if source_min == source_max:
        warnings.append("Source volume is constant; enhancement and noise metrics are limited.")
    return _restore_source_range(working, source_min, source_max), transform_chain, warnings


def _mean_filter(array: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    padded = np.pad(array, [(radius, radius)] * array.ndim, mode="edge")
    output = np.zeros_like(array, dtype=np.float32)
    count = 0
    for offsets in np.ndindex(*(kernel_size for _ in range(array.ndim))):
        slices = tuple(
            slice(offset, offset + size) for offset, size in zip(offsets, array.shape, strict=True)
        )
        output = output + padded[slices]
        count += 1
    return output / float(count)


def _median_filter(array: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    padded = np.pad(array, [(radius, radius)] * array.ndim, mode="edge")
    windows = []
    for offsets in np.ndindex(*(kernel_size for _ in range(array.ndim))):
        slices = tuple(
            slice(offset, offset + size) for offset, size in zip(offsets, array.shape, strict=True)
        )
        windows.append(padded[slices])
    return cast(np.ndarray, np.median(np.stack(windows, axis=0), axis=0).astype(np.float32))


def _percentile_stretch(array: np.ndarray, lower: float, upper: float) -> np.ndarray:
    low = float(np.percentile(array, lower))
    high = float(np.percentile(array, upper))
    if low == high:
        return np.zeros_like(array, dtype=np.float32)
    stretched = np.clip((array - low) / (high - low), 0.0, 1.0)
    return stretched.astype(np.float32)


def _histogram_equalize(array: np.ndarray, bins: int) -> np.ndarray:
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum == maximum:
        return np.zeros_like(array, dtype=np.float32)
    counts, bin_edges = np.histogram(array, bins=bins, range=(minimum, maximum))
    cdf = np.cumsum(counts).astype(np.float64)
    cdf = cdf / cdf[-1]
    equalized = np.interp(array.ravel(), bin_edges[:-1], cdf).reshape(array.shape)
    return cast(np.ndarray, equalized.astype(np.float32))


def _clahe_like(
    array: np.ndarray, *, tiles: tuple[int, int], bins: int, clip_limit: float
) -> np.ndarray:
    slices = array if array.ndim == 3 else array[np.newaxis, ...]
    enhanced = np.stack(
        [
            _clahe_slice(np.asarray(slice_, dtype=np.float32), tiles, bins, clip_limit)
            for slice_ in slices
        ],
        axis=0,
    )
    return enhanced if array.ndim == 3 else enhanced[0]


def _clahe_slice(
    array: np.ndarray, tiles: tuple[int, int], bins: int, clip_limit: float
) -> np.ndarray:
    y_tiles, x_tiles = tiles
    output = np.zeros_like(array, dtype=np.float32)
    y_edges = np.linspace(0, array.shape[0], y_tiles + 1, dtype=int)
    x_edges = np.linspace(0, array.shape[1], x_tiles + 1, dtype=int)
    for y_index in range(y_tiles):
        for x_index in range(x_tiles):
            y0, y1 = y_edges[y_index], y_edges[y_index + 1]
            x0, x1 = x_edges[x_index], x_edges[x_index + 1]
            tile = array[y0:y1, x0:x1]
            output[y0:y1, x0:x1] = _equalize_clipped_tile(tile, bins, clip_limit)
    return output


def _equalize_clipped_tile(tile: np.ndarray, bins: int, clip_limit: float) -> np.ndarray:
    minimum = float(np.min(tile))
    maximum = float(np.max(tile))
    if minimum == maximum:
        return np.zeros_like(tile, dtype=np.float32)
    counts, bin_edges = np.histogram(tile, bins=bins, range=(minimum, maximum))
    limit = max(int(np.ceil(clip_limit * tile.size)), 1)
    excess = np.maximum(counts - limit, 0)
    clipped = np.minimum(counts, limit)
    clipped = clipped + int(np.sum(excess) // bins)
    cdf = np.cumsum(clipped).astype(np.float64)
    cdf = cdf / cdf[-1]
    return cast(
        np.ndarray,
        np.interp(tile.ravel(), bin_edges[:-1], cdf).reshape(tile.shape).astype(np.float32),
    )


def _restore_source_range(array: np.ndarray, source_min: float, source_max: float) -> np.ndarray:
    if source_min == source_max:
        return np.full(array.shape, source_min, dtype=np.float32)
    current_min = float(np.min(array))
    current_max = float(np.max(array))
    if current_min < 0.0 or current_max > 1.0:
        return cast(np.ndarray, np.clip(array, source_min, source_max).astype(np.float32))
    return (array * (source_max - source_min) + source_min).astype(np.float32)
