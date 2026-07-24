"""Technical localisation metrics for synthetic engineering labels."""

from __future__ import annotations

import math

import numpy as np

from medical_imaging_platform.localisation.models import BoundingBox, LocalisationMetrics, Side


def evaluate_side_metrics(
    side: Side,
    predicted_centre: tuple[int, int, int],
    predicted_box: BoundingBox,
    spacing_mm_zyx: tuple[float, float, float],
    *,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
) -> LocalisationMetrics:
    """Evaluate localisation against optional synthetic masks."""
    target = left_mask if side == "left" else right_mask
    opposite = right_mask if side == "left" else left_mask
    if target is None:
        return LocalisationMetrics(
            centre_distance_voxels=None,
            centre_distance_mm=None,
            bounding_box_iou=None,
            target_coverage=None,
            predicted_roi_volume_voxels=_box_volume(predicted_box),
            ground_truth_volume_voxels=None,
            left_right_consistency=True,
            missed_target=None,
            left_right_swap=None,
            evaluation_status="NOT_EVALUATED",
        )
    target_bool = target.astype(bool)
    target_centre = mask_centre_voxel(target_bool)
    if target_centre is None:
        return LocalisationMetrics(
            centre_distance_voxels=None,
            centre_distance_mm=None,
            bounding_box_iou=None,
            target_coverage=None,
            predicted_roi_volume_voxels=_box_volume(predicted_box),
            ground_truth_volume_voxels=0,
            left_right_consistency=True,
            missed_target=True,
            left_right_swap=None,
            evaluation_status="AVAILABLE",
        )
    distance_voxels = float(math.dist(predicted_centre, target_centre))
    distance_mm = float(
        math.dist(
            _scale(predicted_centre, spacing_mm_zyx),
            _scale(target_centre, spacing_mm_zyx),
        )
    )
    predicted_mask = box_mask(predicted_box, target_bool.shape)
    coverage = float(np.count_nonzero(predicted_mask & target_bool) / np.count_nonzero(target_bool))
    iou = mask_iou(predicted_mask, target_bool)
    swap = False
    if opposite is not None and np.any(opposite):
        opposite_centre = mask_centre_voxel(opposite.astype(bool))
        if opposite_centre is not None:
            swap = math.dist(predicted_centre, opposite_centre) < distance_voxels
    return LocalisationMetrics(
        centre_distance_voxels=distance_voxels,
        centre_distance_mm=distance_mm,
        bounding_box_iou=iou,
        target_coverage=coverage,
        predicted_roi_volume_voxels=int(np.count_nonzero(predicted_mask)),
        ground_truth_volume_voxels=int(np.count_nonzero(target_bool)),
        left_right_consistency=not swap,
        missed_target=coverage == 0.0,
        left_right_swap=swap,
        evaluation_status="AVAILABLE",
    )


def mask_centre_voxel(mask: np.ndarray) -> tuple[float, float, float] | None:
    if not np.any(mask):
        return None
    centre = np.mean(np.argwhere(mask), axis=0)
    return (float(centre[0]), float(centre[1]), float(centre[2]))


def mask_iou(first: np.ndarray, second: np.ndarray) -> float | None:
    union = int(np.count_nonzero(first | second))
    if union == 0:
        return None
    return float(np.count_nonzero(first & second) / union)


def box_mask(box: BoundingBox, shape: tuple[int, int, int] | None) -> np.ndarray:
    if shape is None:
        shape = (box.z[1], box.y[1], box.x[1])
    mask = np.zeros(shape, dtype=bool)
    mask[box.z[0] : box.z[1], box.y[0] : box.y[1], box.x[0] : box.x[1]] = True
    return mask


def _box_volume(box: BoundingBox) -> int:
    return int((box.z[1] - box.z[0]) * (box.y[1] - box.y[0]) * (box.x[1] - box.x[0]))


def _scale(
    centre: tuple[float, float, float], spacing_mm_zyx: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        float(centre[0] * spacing_mm_zyx[0]),
        float(centre[1] * spacing_mm_zyx[1]),
        float(centre[2] * spacing_mm_zyx[2]),
    )
