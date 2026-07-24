"""Deterministic lesion matching for previous/current masks."""

from __future__ import annotations

import numpy as np

from medical_imaging_platform.longitudinal.measurements import extract_components
from medical_imaging_platform.longitudinal.models import (
    LesionMatch,
    LesionMeasurement,
    LongitudinalConfig,
)


def match_lesions(
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
    previous_measurements: list[LesionMeasurement],
    current_measurements: list[LesionMeasurement],
    config: LongitudinalConfig,
) -> list[LesionMatch]:
    """Match components greedily with stable tie-breaking and ambiguity detection."""
    previous_components = extract_components(previous_mask, config.minimum_component_voxels)
    current_components = extract_components(current_mask, config.minimum_component_voxels)
    previous = [item for item in previous_measurements if not item.empty_mask]
    current = [item for item in current_measurements if not item.empty_mask]
    if not previous and not current:
        return []
    candidates: list[LesionMatch] = []
    for prev_index, prev in enumerate(previous):
        for curr_index, curr in enumerate(current):
            distance = _centroid_distance(prev, curr)
            overlap_iou = _mask_iou(previous_components[prev_index], current_components[curr_index])
            overlap_dice = _mask_dice(
                previous_components[prev_index], current_components[curr_index]
            )
            box_iou = _box_iou(prev.bounding_box_voxel, curr.bounding_box_voxel)
            distance_term = max(0.0, 1.0 - distance / config.maximum_match_distance_mm)
            overlap_term = max(overlap_iou, overlap_dice, box_iou)
            score = config.centroid_weight * distance_term + config.overlap_weight * overlap_term
            eligible = distance <= config.maximum_match_distance_mm and (
                overlap_iou >= config.minimum_match_iou or overlap_dice >= config.minimum_match_dice
            )
            candidates.append(
                LesionMatch(
                    match_id=f"candidate-{prev.lesion_id}-{curr.lesion_id}",
                    previous_lesion_id=prev.lesion_id,
                    current_lesion_id=curr.lesion_id,
                    centroid_distance_mm=distance,
                    bounding_box_iou=box_iou,
                    overlap_dice=overlap_dice,
                    overlap_iou=overlap_iou,
                    matching_score=float(score if eligible else -1.0),
                    status="matched" if eligible else "ambiguous",
                    ambiguous=False,
                )
            )
    eligible_candidates = sorted(
        [candidate for candidate in candidates if candidate.status == "matched"],
        key=lambda item: (
            -item.matching_score,
            item.previous_lesion_id or "",
            item.current_lesion_id or "",
        ),
    )
    used_previous: set[str] = set()
    used_current: set[str] = set()
    matches: list[LesionMatch] = []
    ambiguous_pairs = _ambiguous_pairs(eligible_candidates)
    for candidate in eligible_candidates:
        if (
            candidate.previous_lesion_id in used_previous
            or candidate.current_lesion_id in used_current
        ):
            continue
        ambiguous = (
            candidate.previous_lesion_id,
            candidate.current_lesion_id,
        ) in ambiguous_pairs
        matches.append(
            candidate.model_copy(
                update={
                    "match_id": f"match-{len(matches) + 1:03d}",
                    "ambiguous": ambiguous,
                    "status": "ambiguous" if ambiguous else "matched",
                }
            )
        )
        used_previous.add(candidate.previous_lesion_id or "")
        used_current.add(candidate.current_lesion_id or "")
    for prev in previous:
        if prev.lesion_id not in used_previous:
            matches.append(_unmatched("previous", prev.lesion_id, len(matches) + 1))
    for curr in current:
        if curr.lesion_id not in used_current:
            matches.append(_unmatched("current", curr.lesion_id, len(matches) + 1))
    return matches


def _centroid_distance(previous: LesionMeasurement, current: LesionMeasurement) -> float:
    if previous.centroid_physical_mm is None or current.centroid_physical_mm is None:
        return float("inf")
    delta = np.asarray(previous.centroid_physical_mm) - np.asarray(current.centroid_physical_mm)
    return float(np.sqrt(np.sum(delta * delta)))


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return 1.0 if union == 0 else float(intersection / union)


def _mask_dice(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    denominator = int(np.count_nonzero(first) + np.count_nonzero(second))
    return 1.0 if denominator == 0 else float((2 * intersection) / denominator)


def _box_iou(first: list[list[int]], second: list[list[int]]) -> float:
    intersection = 1
    first_volume = 1
    second_volume = 1
    for axis in range(3):
        lo = max(first[axis][0], second[axis][0])
        hi = min(first[axis][1], second[axis][1])
        intersection *= max(0, hi - lo)
        first_volume *= max(0, first[axis][1] - first[axis][0])
        second_volume *= max(0, second[axis][1] - second[axis][0])
    union = first_volume + second_volume - intersection
    return 1.0 if union == 0 else float(intersection / union)


def _ambiguous_pairs(candidates: list[LesionMatch]) -> set[tuple[str | None, str | None]]:
    ambiguous: set[tuple[str | None, str | None]] = set()
    by_previous: dict[str | None, list[LesionMatch]] = {}
    by_current: dict[str | None, list[LesionMatch]] = {}
    for candidate in candidates:
        by_previous.setdefault(candidate.previous_lesion_id, []).append(candidate)
        by_current.setdefault(candidate.current_lesion_id, []).append(candidate)
    for grouped in list(by_previous.values()) + list(by_current.values()):
        if len(grouped) > 1:
            top = max(item.matching_score for item in grouped)
            close = [item for item in grouped if abs(item.matching_score - top) <= 1e-9]
            if len(close) > 1:
                ambiguous.update(
                    (item.previous_lesion_id, item.current_lesion_id) for item in close
                )
    return ambiguous


def _unmatched(which: str, lesion_id: str, index: int) -> LesionMatch:
    return LesionMatch(
        match_id=f"match-{index:03d}",
        previous_lesion_id=lesion_id if which == "previous" else None,
        current_lesion_id=lesion_id if which == "current" else None,
        centroid_distance_mm=None,
        bounding_box_iou=None,
        overlap_dice=None,
        overlap_iou=None,
        matching_score=0.0,
        status="unmatched_previous" if which == "previous" else "unmatched_current",
    )
