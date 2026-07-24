"""Slice completeness and ordering checks."""

from __future__ import annotations

import numpy as np

from medical_imaging_platform.ingestion.models import OrderedSliceSet
from medical_imaging_platform.quality_control.models import QualityControlConfig, QualityFinding
from medical_imaging_platform.quality_control.rules import rule_info


def run_slice_checks(
    ordered: OrderedSliceSet,
    config: QualityControlConfig,
) -> tuple[list[QualityFinding], dict[str, object]]:
    """Run series-level slice completeness checks."""
    metadata = ordered.files
    findings: list[QualityFinding] = []
    positions = [item.image_position_patient[2] for item in metadata if item.image_position_patient]
    instance_numbers = [
        item.instance_number for item in metadata if item.instance_number is not None
    ]
    gaps = [b - a for a, b in zip(sorted(positions), sorted(positions)[1:], strict=False)]
    median_gap = float(np.median(gaps)) if gaps else None
    irregular_gaps = [
        gap
        for gap in gaps
        if median_gap is not None and abs(gap - median_gap) > config.slice_gap_tolerance
    ]

    _append(
        findings,
        "DICOM-QC-ORD-001",
        "FAIL"
        if any(issue.severity in {"ERROR", "CRITICAL"} for issue in ordered.issues)
        else "PASS",
        f"Ordering strategy: {ordered.strategy}",
        [item.file_path for item in metadata],
        ordered.strategy,
        "deterministic ordering",
        "Review DICOM ordering metadata.",
    )
    rounded_positions = {round(value, 4) for value in positions}
    if len(rounded_positions) != len(positions):
        _append(
            findings,
            "DICOM-QC-SLC-001",
            "FAIL",
            "Duplicate slice positions detected.",
            [item.file_path for item in metadata],
            len(positions) - len(rounded_positions),
            0,
            "Review duplicate ImagePositionPatient values.",
        )
    if len(set(instance_numbers)) != len(instance_numbers):
        _append(
            findings,
            "DICOM-QC-SLC-003",
            "FAIL",
            "Duplicate instance numbers detected.",
            [item.file_path for item in metadata],
            len(instance_numbers) - len(set(instance_numbers)),
            0,
            "Review duplicate InstanceNumber values.",
        )
    if irregular_gaps:
        _append(
            findings,
            "DICOM-QC-SLC-002",
            "FAIL",
            "Irregular slice spacing detected; missing slices may only be inferred "
            "with reliable spacing.",
            [item.file_path for item in metadata],
            irregular_gaps,
            f"within {config.slice_gap_tolerance}",
            "Review ordered slice positions.",
        )
    elif len(metadata) < config.minimum_slice_count:
        _append(
            findings,
            "DICOM-QC-SLC-002",
            "FAIL",
            "Observed slice count is below configured engineering minimum.",
            [item.file_path for item in metadata],
            len(metadata),
            config.minimum_slice_count,
            "Review whether the series is complete.",
        )

    metrics: dict[str, object] = {
        "observed_slice_count": len(metadata),
        "unique_slice_position_count": len(rounded_positions),
        "duplicate_instance_number_count": len(instance_numbers) - len(set(instance_numbers)),
        "median_inter_slice_spacing": median_gap,
        "minimum_inter_slice_spacing": min(gaps) if gaps else None,
        "maximum_inter_slice_spacing": max(gaps) if gaps else None,
        "irregular_spacing_count": len(irregular_gaps),
        "completeness_note": (
            "Missing slices are inferred only when spacing metadata are regular enough."
        ),
    }
    return findings, metrics


def _append(
    findings: list[QualityFinding],
    rule_id: str,
    status: str,
    message: str,
    affected_files: list[str],
    observed_value: object,
    expected_value: object,
    remediation: str,
) -> None:
    category, severity, _ = rule_info(rule_id)
    findings.append(
        QualityFinding(
            rule_id=rule_id,
            category=category,
            severity=severity,
            status=status,  # type: ignore[arg-type]
            message=message,
            affected_files=affected_files,
            observed_value=observed_value,
            expected_value=expected_value,
            remediation=remediation,
        )
    )
