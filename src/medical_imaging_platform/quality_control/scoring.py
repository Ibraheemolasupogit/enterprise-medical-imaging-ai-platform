"""Transparent quality scoring."""

from __future__ import annotations

from medical_imaging_platform.quality_control.models import (
    QualityControlConfig,
    QualityFinding,
    SeriesQualityStatus,
)


def score_findings(
    findings: list[QualityFinding],
    config: QualityControlConfig,
) -> tuple[int, SeriesQualityStatus, dict[str, int]]:
    """Score findings with traceable configured deductions."""
    deductions: dict[str, int] = {}
    score = 100
    failed = [finding for finding in findings if finding.status == "FAIL"]
    for finding in failed:
        deduction = config.scoring_weights.get(finding.severity, 0)
        deductions[finding.rule_id] = deductions.get(finding.rule_id, 0) + deduction
        score -= deduction
    score = max(0, min(100, score))

    failed_rule_ids = {finding.rule_id for finding in failed}
    if any(rule_id in failed_rule_ids for rule_id in config.critical_rule_ids) or any(
        finding.severity == "CRITICAL" and finding.status == "FAIL" for finding in findings
    ):
        return score, "REJECTED", deductions
    if any(finding.severity == "ERROR" and finding.status == "FAIL" for finding in findings):
        return score, "FAIL", deductions
    if any(finding.severity == "WARNING" and finding.status == "FAIL" for finding in findings):
        return score, "PASS_WITH_WARNINGS", deductions
    return score, "PASS", deductions
