"""Localisation quality rules."""

from __future__ import annotations

from medical_imaging_platform.localisation.atlas import left_right_distance_mm
from medical_imaging_platform.localisation.models import (
    LocalisationConfig,
    LocalisationFinding,
    LocalisationMetrics,
    LocalisationStatus,
    Side,
    SideLocalisation,
)


def evaluate_quality(
    left: SideLocalisation,
    right: SideLocalisation,
    metrics: dict[Side, LocalisationMetrics],
    *,
    spacing_mm_zyx: tuple[float, float, float],
    config: LocalisationConfig,
    input_findings: list[LocalisationFinding],
) -> tuple[LocalisationStatus, list[LocalisationFinding]]:
    """Evaluate localisation quality gates."""
    findings = list(input_findings)
    separation = left_right_distance_mm(
        left.predicted_centre_voxel, right.predicted_centre_voxel, spacing_mm_zyx
    )
    if separation < config.left_right_minimum_separation_mm:
        findings.append(_finding("LOC-QC-LR-001", "ERROR", "Left and right centres are too close."))
    for side_output in (left, right):
        findings.extend(_side_findings(side_output, metrics[side_output.side], config))
    if any(finding.severity == "CRITICAL" and finding.status == "FAIL" for finding in findings):
        return "REJECTED", findings
    if any(finding.severity == "ERROR" and finding.status == "FAIL" for finding in findings):
        return "FAILED", findings
    if any(finding.severity == "WARNING" and finding.status == "FAIL" for finding in findings):
        return "LOCALISED_WITH_WARNINGS", findings
    return "LOCALISED", findings


def _side_findings(
    side_output: SideLocalisation,
    metrics: LocalisationMetrics,
    config: LocalisationConfig,
) -> list[LocalisationFinding]:
    findings: list[LocalisationFinding] = []
    if any(size <= 0 for size in side_output.roi_shape):
        findings.append(_finding("LOC-QC-ROI-001", "ERROR", "ROI is empty.", side_output.side))
    if side_output.roi_extraction.padding_fraction > config.maximum_padding_fraction:
        findings.append(
            _finding(
                "LOC-QC-PAD-001",
                "WARNING",
                "ROI required excessive boundary padding.",
                side_output.side,
                side_output.roi_extraction.padding_fraction,
                config.maximum_padding_fraction,
            )
        )
    if metrics.evaluation_status == "NOT_EVALUATED":
        findings.append(
            LocalisationFinding(
                rule_id="LOC-QC-CEN-001",
                severity="INFO",
                status="NOT_EVALUATED",
                message="Optional synthetic ground truth was not provided.",
                side=side_output.side,
                remediation="Provide synthetic masks for engineering metric evaluation.",
            )
        )
        return findings
    if metrics.left_right_swap:
        findings.append(
            _finding("LOC-QC-SWP-001", "ERROR", "Left/right swap detected.", side_output.side)
        )
    if (
        metrics.centre_distance_mm is not None
        and metrics.centre_distance_mm > config.maximum_centre_distance_mm
    ):
        findings.append(
            _finding(
                "LOC-QC-CEN-001",
                "ERROR",
                "Synthetic centre distance exceeds threshold.",
                side_output.side,
                metrics.centre_distance_mm,
                config.maximum_centre_distance_mm,
            )
        )
    if (
        metrics.target_coverage is not None
        and metrics.target_coverage < config.minimum_target_coverage
    ):
        findings.append(
            _finding(
                "LOC-QC-COV-001",
                "ERROR",
                "Synthetic target coverage is below threshold.",
                side_output.side,
                metrics.target_coverage,
                config.minimum_target_coverage,
            )
        )
    if metrics.bounding_box_iou is not None and metrics.bounding_box_iou < config.minimum_box_iou:
        findings.append(
            _finding(
                "LOC-QC-BND-001",
                "ERROR",
                "Synthetic bounding-box IoU is below threshold.",
                side_output.side,
                metrics.bounding_box_iou,
                config.minimum_box_iou,
            )
        )
    return findings


def _finding(
    rule_id: str,
    severity: str,
    message: str,
    side: str | None = None,
    observed: object | None = None,
    expected: object | None = None,
) -> LocalisationFinding:
    return LocalisationFinding(
        rule_id=rule_id,
        severity=severity,  # type: ignore[arg-type]
        status="FAIL",
        message=message,
        side=side,  # type: ignore[arg-type]
        observed_value=observed,
        expected_value=expected,
        remediation="Review localisation input, configuration, and synthetic labels.",
    )
