"""Quality gates for synthetic classification experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from medical_imaging_platform.classification.models import (
    ClassificationConfig,
    ClassificationFinding,
    ClassificationStatus,
)


def evaluate_quality(
    *,
    config: ClassificationConfig,
    training_history: list[dict[str, float | int]],
    evaluation: dict[str, Any],
    checkpoint_paths: list[Path],
    evidence_paths: list[Path],
    leakage_detected: bool = False,
) -> tuple[ClassificationStatus, list[ClassificationFinding]]:
    findings: list[ClassificationFinding] = []
    findings.append(_finding("CLS-QC-DATA-001", "INFO", "Dataset manifest validated.", "PASS"))
    findings.append(
        _finding(
            "CLS-QC-SPLIT-001",
            "CRITICAL" if leakage_detected else "INFO",
            "Subject leakage detected." if leakage_detected else "No subject leakage detected.",
            "FAIL" if leakage_detected else "PASS",
        )
    )
    completed = bool(training_history)
    finite = all(
        math.isfinite(float(row["train_loss"])) and math.isfinite(float(row["validation_loss"]))
        for row in training_history
    )
    findings.append(
        _finding(
            "CLS-QC-TRAIN-001",
            "INFO" if completed and finite else "ERROR",
            "Training completed with finite losses.",
            "PASS" if completed and finite else "FAIL",
        )
    )
    validation = evaluation["validation"]["metrics"]
    test = evaluation["test"]["metrics"]
    findings.extend(
        [
            _minimum("CLS-QC-AUC-001", validation["auroc"], config.minimum_validation_auroc),
            _minimum("CLS-QC-PRC-001", validation["auprc"], config.minimum_validation_auprc),
            _minimum("CLS-QC-REC-001", validation["recall"], config.minimum_validation_recall),
            _maximum("CLS-QC-BRI-001", validation["brier_score"], config.maximum_brier_score),
            _maximum("CLS-QC-FN-001", test["false_negative_count"], config.maximum_false_negatives),
        ]
    )
    findings.append(
        _finding(
            "CLS-QC-CAL-001",
            "INFO",
            "Calibration evidence recorded.",
            "PASS" if evaluation.get("calibration_present") else "FAIL",
        )
    )
    findings.append(
        _finding(
            "CLS-QC-THR-001",
            "INFO",
            "Threshold evidence recorded from validation split.",
            "PASS" if evaluation.get("threshold_present") else "FAIL",
        )
    )
    findings.append(
        _finding(
            "CLS-QC-ABS-001",
            "INFO" if config.abstention_lower <= config.abstention_upper else "ERROR",
            "Abstention interval configured.",
            "PASS" if config.abstention_lower <= config.abstention_upper else "FAIL",
        )
    )
    checkpoints_ok = all(path.exists() for path in checkpoint_paths)
    evidence_ok = all(path.exists() for path in evidence_paths)
    findings.append(
        _finding(
            "CLS-QC-CHK-001",
            "INFO" if checkpoints_ok and evidence_ok else "ERROR",
            "Checkpoint and evidence files are complete.",
            "PASS" if checkpoints_ok and evidence_ok else "FAIL",
        )
    )
    if any(finding.severity == "CRITICAL" and finding.status == "FAIL" for finding in findings):
        return "REJECTED", findings
    if any(finding.severity == "ERROR" and finding.status == "FAIL" for finding in findings):
        return "FAIL", findings
    if any(finding.severity == "WARNING" and finding.status == "FAIL" for finding in findings):
        return "PASS_WITH_WARNINGS", findings
    return "PASS", findings


def _minimum(rule_id: str, observed: object, expected: float) -> ClassificationFinding:
    value = _to_float(observed)
    return _finding(
        rule_id,
        "ERROR" if value < expected else "INFO",
        "Metric meets configured synthetic threshold."
        if value >= expected
        else "Metric is below configured synthetic threshold.",
        "PASS" if value >= expected else "FAIL",
        value,
        expected,
    )


def _maximum(rule_id: str, observed: object, expected: float) -> ClassificationFinding:
    value = _to_float(observed)
    return _finding(
        rule_id,
        "ERROR" if value > expected else "INFO",
        "Metric is within configured synthetic threshold."
        if value <= expected
        else "Metric exceeds configured synthetic threshold.",
        "PASS" if value <= expected else "FAIL",
        value,
        expected,
    )


def _finding(
    rule_id: str,
    severity: str,
    message: str,
    status: str,
    observed: object | None = None,
    expected: object | None = None,
) -> ClassificationFinding:
    return ClassificationFinding(
        rule_id=rule_id,
        severity=severity,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        message=message,
        observed_value=observed,
        expected_value=expected,
        remediation="Review classification dataset, calibration, threshold policy, and evidence.",
    )


def _to_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"Expected numeric classification metric, got {type(value).__name__}")
