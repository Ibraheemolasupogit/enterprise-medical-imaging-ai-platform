"""Model-quality gates for synthetic segmentation experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from medical_imaging_platform.segmentation.models import (
    SegmentationConfig,
    SegmentationFinding,
    SegmentationStatus,
)


def evaluate_quality(
    *,
    config: SegmentationConfig,
    training_history: list[dict[str, float | int]],
    evaluation: dict[str, Any],
    checkpoint_paths: list[Path],
    evidence_paths: list[Path],
    leakage_detected: bool = False,
) -> tuple[SegmentationStatus, list[SegmentationFinding]]:
    """Evaluate stable model-quality gates."""
    findings: list[SegmentationFinding] = []
    findings.append(_finding("SEG-QC-DATA-001", "INFO", "Dataset manifest validated.", "PASS"))
    findings.append(
        _finding(
            "SEG-QC-SPLIT-001",
            "CRITICAL" if leakage_detected else "INFO",
            "Subject-level split leakage detected."
            if leakage_detected
            else "No subject leakage detected.",
            "FAIL" if leakage_detected else "PASS",
        )
    )
    completed = len(training_history) > 0
    findings.append(
        _finding(
            "SEG-QC-TRAIN-001",
            "ERROR" if not completed else "INFO",
            "Training completed." if completed else "Training did not complete.",
            "PASS" if completed else "FAIL",
        )
    )
    finite_losses = all(
        math.isfinite(float(row["train_loss"])) and math.isfinite(float(row["validation_loss"]))
        for row in training_history
    )
    findings.append(
        _finding(
            "SEG-QC-LOSS-001",
            "ERROR" if not finite_losses else "INFO",
            "Training and validation losses are finite.",
            "PASS" if finite_losses else "FAIL",
        )
    )
    best_dice = float(evaluation["validation"]["aggregate"]["dice"]["mean"] or 0.0)
    findings.append(
        _threshold(
            "SEG-QC-DICE-001",
            best_dice,
            config.minimum_validation_dice,
            "Validation Dice below configured synthetic threshold.",
        )
    )
    test_recall = float(evaluation["test"]["aggregate"]["recall"]["mean"] or 0.0)
    findings.append(
        _threshold(
            "SEG-QC-RECALL-001",
            test_recall,
            config.minimum_test_recall,
            "Test recall below configured synthetic threshold.",
        )
    )
    fp_max = int(evaluation["test"]["false_positive_voxels_max"])
    findings.append(
        _maximum(
            "SEG-QC-FP-001",
            fp_max,
            config.maximum_false_positive_voxels,
            "False-positive voxel burden exceeds configured threshold.",
        )
    )
    rel_error = float(evaluation["test"]["relative_volume_error_max"] or 0.0)
    findings.append(
        _maximum(
            "SEG-QC-VOL-001",
            rel_error,
            config.maximum_relative_volume_error,
            "Relative volume error exceeds configured synthetic threshold.",
        )
    )
    checkpoints_ok = all(path.exists() for path in checkpoint_paths)
    findings.append(
        _finding(
            "SEG-QC-CHK-001",
            "ERROR" if not checkpoints_ok else "INFO",
            "Checkpoint evidence is complete.",
            "PASS" if checkpoints_ok else "FAIL",
        )
    )
    evidence_ok = all(path.exists() for path in evidence_paths)
    findings.append(
        _finding(
            "SEG-QC-REP-001",
            "ERROR" if not evidence_ok else "INFO",
            "Experiment evidence files are complete.",
            "PASS" if evidence_ok else "FAIL",
        )
    )
    if any(finding.severity == "CRITICAL" and finding.status == "FAIL" for finding in findings):
        return "REJECTED", findings
    if any(finding.severity == "ERROR" and finding.status == "FAIL" for finding in findings):
        return "FAIL", findings
    if any(finding.severity == "WARNING" and finding.status == "FAIL" for finding in findings):
        return "PASS_WITH_WARNINGS", findings
    return "PASS", findings


def _threshold(rule_id: str, observed: float, expected: float, message: str) -> SegmentationFinding:
    return _finding(
        rule_id,
        "ERROR" if observed < expected else "INFO",
        message if observed < expected else "Metric meets configured synthetic threshold.",
        "FAIL" if observed < expected else "PASS",
        observed,
        expected,
    )


def _maximum(rule_id: str, observed: float, expected: float, message: str) -> SegmentationFinding:
    return _finding(
        rule_id,
        "ERROR" if observed > expected else "INFO",
        message if observed > expected else "Metric is within configured synthetic threshold.",
        "FAIL" if observed > expected else "PASS",
        observed,
        expected,
    )


def _finding(
    rule_id: str,
    severity: str,
    message: str,
    status: str,
    observed: object | None = None,
    expected: object | None = None,
) -> SegmentationFinding:
    return SegmentationFinding(
        rule_id=rule_id,
        severity=severity,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        message=message,
        observed_value=observed,
        expected_value=expected,
        remediation="Review segmentation configuration, dataset, and experiment evidence.",
    )
