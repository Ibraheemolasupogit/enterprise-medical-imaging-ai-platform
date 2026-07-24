"""Binary segmentation metrics for synthetic lesion masks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from medical_imaging_platform.segmentation.models import SegmentationMetricSet


def compute_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
) -> SegmentationMetricSet:
    """Compute voxel and spacing-aware surface metrics."""
    pred = np.asarray(prediction).astype(bool)
    gt = np.asarray(target).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError("Prediction and target shapes must match.")
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    tn = int(np.count_nonzero(~pred & ~gt))
    pred_count = int(np.count_nonzero(pred))
    gt_count = int(np.count_nonzero(gt))
    both_empty = pred_count == 0 and gt_count == 0
    dice = 1.0 if both_empty else _safe_divide(2 * tp, 2 * tp + fp + fn)
    iou = 1.0 if both_empty else _safe_divide(tp, tp + fp + fn)
    precision = 1.0 if pred_count == 0 and gt_count == 0 else _safe_divide(tp, tp + fp)
    recall = 1.0 if gt_count == 0 and pred_count == 0 else _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    rel_error = None if gt_count == 0 else abs(pred_count - gt_count) / gt_count
    hd95, asd = _surface_metrics(pred, gt, spacing_mm)
    return SegmentationMetricSet(
        dice=dice,
        iou=iou,
        precision=precision,
        recall=recall,
        sensitivity=recall,
        specificity=specificity,
        false_positive_voxels=fp,
        false_negative_voxels=fn,
        predicted_lesion_volume_voxels=pred_count,
        ground_truth_lesion_volume_voxels=gt_count,
        absolute_volume_error_voxels=abs(pred_count - gt_count),
        relative_volume_error=rel_error,
        hausdorff95_mm=hd95,
        average_surface_distance_mm=asd,
        both_masks_empty=both_empty,
    )


def aggregate_metrics(metrics: list[SegmentationMetricSet]) -> dict[str, dict[str, float | None]]:
    """Aggregate numeric metrics with mean, median, std, min, and max."""
    keys = ["dice", "iou", "precision", "recall", "specificity", "relative_volume_error"]
    output: dict[str, dict[str, float | None]] = {}
    for key in keys:
        values = [
            float(value)
            for item in metrics
            if (value := getattr(item, key)) is not None and np.isfinite(value)
        ]
        output[key] = _summary(values)
    return output


def _surface_metrics(
    prediction: np.ndarray, target: np.ndarray, spacing_mm: tuple[float, float, float]
) -> tuple[float | None, float | None]:
    if not np.any(prediction) and not np.any(target):
        return 0.0, 0.0
    if not np.any(prediction) or not np.any(target):
        return None, None
    pred_surface = prediction ^ ndimage.binary_erosion(prediction)
    target_surface = target ^ ndimage.binary_erosion(target)
    pred_dist = ndimage.distance_transform_edt(~pred_surface, sampling=spacing_mm)
    target_dist = ndimage.distance_transform_edt(~target_surface, sampling=spacing_mm)
    distances = np.concatenate([target_dist[pred_surface], pred_dist[target_surface]])
    if distances.size == 0:
        return 0.0, 0.0
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
