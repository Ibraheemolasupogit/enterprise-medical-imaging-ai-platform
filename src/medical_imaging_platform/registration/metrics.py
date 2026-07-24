"""Technical registration metrics."""

from __future__ import annotations

import math
import time

import numpy as np
import SimpleITK as sitk  # noqa: N813

from medical_imaging_platform.registration.models import RegistrationMetrics


def metric_timer() -> float:
    return time.perf_counter()


def compute_metrics(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
    foreground_threshold: float,
) -> RegistrationMetrics:
    """Compute deterministic technical metrics for two [z, y, x] arrays."""
    fixed_f = fixed.astype(np.float64)
    moving_f = moving.astype(np.float64)
    mse = float(np.mean((fixed_f - moving_f) ** 2))
    ncc = normalised_cross_correlation(fixed_f, moving_f)
    mi = mutual_information(fixed_f, moving_f)
    fixed_mask = np.abs(fixed_f) > foreground_threshold
    moving_mask = np.abs(moving_f) > foreground_threshold
    dice = dice_coefficient(fixed_mask, moving_mask)
    overlap = foreground_overlap(fixed_mask, moving_mask)
    com_distance = centre_of_mass_distance(fixed_mask, moving_mask, spacing_zyx)
    return RegistrationMetrics(
        mean_squared_error=mse,
        normalised_cross_correlation=ncc,
        mutual_information=mi,
        foreground_overlap=overlap,
        dice_coefficient=dice,
        centre_of_mass_distance_mm=com_distance,
    )


def normalised_cross_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = first.ravel() - float(np.mean(first))
    second_flat = second.ravel() - float(np.mean(second))
    denominator = float(np.linalg.norm(first_flat) * np.linalg.norm(second_flat))
    if denominator == 0:
        return 0.0
    return float(np.dot(first_flat, second_flat) / denominator)


def mutual_information(first: np.ndarray, second: np.ndarray, bins: int = 32) -> float:
    histogram, _, _ = np.histogram2d(first.ravel(), second.ravel(), bins=bins)
    total = float(np.sum(histogram))
    if total == 0:
        return 0.0
    probability = histogram / total
    px = np.sum(probability, axis=1)
    py = np.sum(probability, axis=0)
    px_py = px[:, None] * py[None, :]
    nonzero = probability > 0
    return float(np.sum(probability[nonzero] * np.log(probability[nonzero] / px_py[nonzero])))


def dice_coefficient(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = int(np.count_nonzero(first) + np.count_nonzero(second))
    if denominator == 0:
        return None
    return float(2 * np.count_nonzero(first & second) / denominator)


def foreground_overlap(first: np.ndarray, second: np.ndarray) -> float | None:
    union = int(np.count_nonzero(first | second))
    if union == 0:
        return None
    return float(np.count_nonzero(first & second) / union)


def centre_of_mass_distance(
    fixed_mask: np.ndarray, moving_mask: np.ndarray, spacing_zyx: tuple[float, float, float]
) -> float | None:
    fixed_com = mask_centre_of_mass(fixed_mask, spacing_zyx)
    moving_com = mask_centre_of_mass(moving_mask, spacing_zyx)
    if fixed_com is None or moving_com is None:
        return None
    return float(math.dist(fixed_com, moving_com))


def mask_centre_of_mass(
    mask: np.ndarray, spacing_zyx: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    if not np.any(mask):
        return None
    indices = np.argwhere(mask)
    mean_zyx = np.mean(indices, axis=0)
    return (
        float(mean_zyx[2] * spacing_zyx[2]),
        float(mean_zyx[1] * spacing_zyx[1]),
        float(mean_zyx[0] * spacing_zyx[0]),
    )


def sitk_metric_value(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    transform: sitk.Transform,
    metric: str,
    bins: int,
) -> float:
    registration = sitk.ImageRegistrationMethod()
    if metric == "normalised_correlation":
        registration.SetMetricAsCorrelation()
    else:
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    registration.SetInitialTransform(transform, inPlace=False)
    return float(registration.MetricEvaluate(fixed_image, moving_image))
