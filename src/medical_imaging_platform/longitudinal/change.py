"""Longitudinal change calculations and engineering labels."""

from __future__ import annotations

import math

from medical_imaging_platform.longitudinal.models import (
    LesionMatch,
    LesionMeasurement,
    LongitudinalChange,
    LongitudinalConfig,
    LongitudinalLabel,
)


def calculate_changes(
    matches: list[LesionMatch],
    previous_measurements: list[LesionMeasurement],
    current_measurements: list[LesionMeasurement],
    config: LongitudinalConfig,
    *,
    force_indeterminate_reasons: list[str] | None = None,
) -> list[LongitudinalChange]:
    """Calculate change metrics and synthetic engineering labels."""
    previous_by_id = {item.lesion_id: item for item in previous_measurements if not item.empty_mask}
    current_by_id = {item.lesion_id: item for item in current_measurements if not item.empty_mask}
    forced = force_indeterminate_reasons or []
    if not matches and forced:
        return [
            LongitudinalChange(
                change_id="change-001",
                label="indeterminate",
                previous_lesion_id=None,
                current_lesion_id=None,
                absolute_volume_change_mm3=None,
                percentage_volume_change=None,
                absolute_diameter_change_mm=None,
                percentage_diameter_change=None,
                centroid_displacement_mm=None,
                overlap_dice=None,
                overlap_iou=None,
                reasons=forced,
            )
        ]
    changes: list[LongitudinalChange] = []
    for index, match in enumerate(matches, start=1):
        previous = previous_by_id.get(match.previous_lesion_id or "")
        current = current_by_id.get(match.current_lesion_id or "")
        reasons = list(forced)
        label: LongitudinalLabel
        volume_abs: float | None = None
        volume_pct: float | None = None
        diameter_abs: float | None = None
        diameter_pct: float | None = None
        if match.ambiguous or match.status == "ambiguous":
            reasons.append("ambiguous lesion match")
            label = "indeterminate"
        elif forced:
            label = "indeterminate"
        elif previous is None and current is not None:
            label = "new"
            reasons.append("current lesion unmatched")
        elif previous is not None and current is None:
            label = "resolved"
            reasons.append("previous lesion unmatched")
        elif previous is None or current is None:
            label = "indeterminate"
            reasons.append("missing lesion measurement")
        else:
            volume_abs = current.physical_volume_mm3 - previous.physical_volume_mm3
            volume_pct = _percent_change(
                previous.physical_volume_mm3,
                current.physical_volume_mm3,
                config.small_denominator_mm3,
            )
            diameter_abs = current.maximum_3d_diameter_mm - previous.maximum_3d_diameter_mm
            diameter_pct = _percent_change(
                previous.maximum_3d_diameter_mm,
                current.maximum_3d_diameter_mm,
                max(math.sqrt(config.small_denominator_mm3), 1e-6),
            )
            label = _label_matched(volume_pct, diameter_pct, config)
            reasons.append("matched lesion change within synthetic engineering thresholds")
        changes.append(
            LongitudinalChange(
                change_id=f"change-{index:03d}",
                label=label,
                previous_lesion_id=match.previous_lesion_id,
                current_lesion_id=match.current_lesion_id,
                absolute_volume_change_mm3=volume_abs,
                percentage_volume_change=volume_pct,
                absolute_diameter_change_mm=diameter_abs,
                percentage_diameter_change=diameter_pct,
                centroid_displacement_mm=match.centroid_distance_mm,
                overlap_dice=match.overlap_dice,
                overlap_iou=match.overlap_iou,
                reasons=reasons,
            )
        )
    return changes


def _percent_change(previous: float, current: float, small_denominator: float) -> float | None:
    if previous < small_denominator:
        return None
    return float(((current - previous) / previous) * 100.0)


def _label_matched(
    volume_pct: float | None, diameter_pct: float | None, config: LongitudinalConfig
) -> LongitudinalLabel:
    increased = (
        volume_pct is not None and volume_pct >= config.volume_increase_threshold_percent
    ) or (diameter_pct is not None and diameter_pct >= config.diameter_increase_threshold_percent)
    reduced = (
        volume_pct is not None and volume_pct <= -config.volume_reduction_threshold_percent
    ) or (diameter_pct is not None and diameter_pct <= -config.diameter_reduction_threshold_percent)
    if increased and reduced:
        return "indeterminate"
    if increased:
        return "increased"
    if reduced:
        return "reduced"
    return "stable"
