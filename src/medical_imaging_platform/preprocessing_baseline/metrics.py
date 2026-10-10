"""Deterministic image-content metrics for scalar medical-imaging arrays."""

from __future__ import annotations

import numpy as np

from medical_imaging_platform.preprocessing_baseline.models import (
    ImageContentMetrics,
    MetricSettings,
)


def compute_image_content_metrics(
    array: np.ndarray, settings: MetricSettings | None = None
) -> ImageContentMetrics:
    """Compute deterministic image-content QC metrics for a 2D slice or 3D volume."""
    metric_settings = settings or MetricSettings()
    volume = validate_numeric_array(array)
    percentiles = {
        _percentile_key(item): float(np.percentile(volume, item))
        for item in metric_settings.percentiles
    }
    laplacian = _laplacian(volume)
    gradient_magnitude = _gradient_magnitude(volume)
    edge_threshold = float(np.percentile(gradient_magnitude, metric_settings.edge_percentile))
    if edge_threshold == 0.0:
        edge_density = 0.0
    else:
        edge_density = float(np.count_nonzero(gradient_magnitude > edge_threshold) / volume.size)
    residual = volume - _mean_filter(volume, metric_settings.local_contrast_kernel_size)
    return ImageContentMetrics(
        shape=tuple(int(item) for item in volume.shape),
        dtype=str(volume.dtype),
        minimum=float(np.min(volume)),
        maximum=float(np.max(volume)),
        mean=float(np.mean(volume)),
        standard_deviation=float(np.std(volume)),
        percentiles=percentiles,
        sharpness_laplacian_variance=float(np.var(laplacian)),
        noise_high_frequency_residual_std=float(np.std(residual)),
        edge_density=edge_density,
        entropy=_entropy(volume, metric_settings.histogram_bins),
        local_contrast_mean=float(np.mean(np.abs(residual))),
    )


def validate_numeric_array(array: np.ndarray) -> np.ndarray:
    """Return a float32 copy after validating array dimensionality and finite values."""
    volume = np.asarray(array)
    if volume.ndim not in {2, 3}:
        raise ValueError("image-content metrics require a 2D slice or 3D volume")
    if volume.size == 0:
        raise ValueError("image-content metrics require a non-empty array")
    if not np.issubdtype(volume.dtype, np.number):
        raise ValueError("image-content metrics require a numeric array")
    output = volume.astype(np.float32, copy=True)
    if not np.all(np.isfinite(output)):
        raise ValueError("image-content metrics require finite numeric values")
    return output


def metric_deltas(before: ImageContentMetrics, after: ImageContentMetrics) -> dict[str, float]:
    """Return stable after-minus-before deltas for scalar metric fields."""
    return {
        "minimum": after.minimum - before.minimum,
        "maximum": after.maximum - before.maximum,
        "mean": after.mean - before.mean,
        "standard_deviation": after.standard_deviation - before.standard_deviation,
        "sharpness_laplacian_variance": (
            after.sharpness_laplacian_variance - before.sharpness_laplacian_variance
        ),
        "noise_high_frequency_residual_std": (
            after.noise_high_frequency_residual_std - before.noise_high_frequency_residual_std
        ),
        "edge_density": after.edge_density - before.edge_density,
        "entropy": after.entropy - before.entropy,
        "local_contrast_mean": after.local_contrast_mean - before.local_contrast_mean,
    }


def _laplacian(array: np.ndarray) -> np.ndarray:
    padded = _edge_pad(array, 1)
    center = _center_view(padded, array.ndim, 1) * (-2 * array.ndim)
    result = center.copy()
    for axis in range(array.ndim):
        before = _offset_view(padded, array.ndim, axis, 0)
        after = _offset_view(padded, array.ndim, axis, 2)
        result = result + before + after
    return result


def _gradient_magnitude(array: np.ndarray) -> np.ndarray:
    gradients = np.gradient(array)
    squared = np.zeros_like(array, dtype=np.float32)
    for gradient in gradients:
        squared = squared + np.asarray(gradient, dtype=np.float32) ** 2
    return np.sqrt(squared)


def _mean_filter(array: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    padded = _edge_pad(array, radius)
    output = np.zeros_like(array, dtype=np.float32)
    count = 0
    for offsets in np.ndindex(*(kernel_size for _ in range(array.ndim))):
        slices = tuple(
            slice(offset, offset + size) for offset, size in zip(offsets, array.shape, strict=True)
        )
        output = output + padded[slices]
        count += 1
    return output / float(count)


def _entropy(array: np.ndarray, bins: int) -> float:
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum == maximum:
        return 0.0
    counts, _ = np.histogram(array, bins=bins, range=(minimum, maximum))
    probabilities = counts.astype(np.float64)
    probabilities = probabilities[probabilities > 0]
    probabilities = probabilities / float(np.sum(probabilities))
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _edge_pad(array: np.ndarray, radius: int) -> np.ndarray:
    return np.pad(array, [(radius, radius)] * array.ndim, mode="edge")


def _center_view(padded: np.ndarray, ndim: int, radius: int) -> np.ndarray:
    return padded[tuple(slice(radius, -radius) for _ in range(ndim))]


def _offset_view(padded: np.ndarray, ndim: int, axis: int, offset: int) -> np.ndarray:
    slices = [slice(1, -1) for _ in range(ndim)]
    slices[axis] = slice(offset, -2 + offset if offset < 2 else None)
    return padded[tuple(slices)]


def _percentile_key(value: float) -> str:
    text = f"{value:g}".replace(".", "_")
    return f"p{text}"
