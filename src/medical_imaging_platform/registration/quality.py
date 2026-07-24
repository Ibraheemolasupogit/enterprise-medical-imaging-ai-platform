"""Registration plausibility and quality gates."""

from __future__ import annotations

import math

import numpy as np

from medical_imaging_platform.registration.models import (
    RegistrationConfig,
    RegistrationFinding,
    RegistrationMetrics,
    RegistrationStatus,
    TransformSummary,
)


def evaluate_registration_quality(
    transform: TransformSummary,
    before: RegistrationMetrics,
    after: RegistrationMetrics,
    registered: np.ndarray,
    *,
    default_pixel_value: float,
    config: RegistrationConfig,
    precondition_findings: list[RegistrationFinding],
) -> tuple[RegistrationStatus, list[RegistrationFinding]]:
    """Apply deterministic registration quality rules."""
    findings = list(precondition_findings)
    findings.extend(_transform_findings(transform, config))
    findings.extend(_metric_findings(before, after, config))
    findings.extend(_padding_findings(registered, default_pixel_value, config))

    if any(finding.severity == "CRITICAL" and finding.status == "FAIL" for finding in findings):
        return "REJECTED", findings
    if any(finding.severity == "ERROR" and finding.status == "FAIL" for finding in findings):
        return "FAIL", findings
    if any(finding.severity == "WARNING" and finding.status == "FAIL" for finding in findings):
        return "PASS_WITH_WARNINGS", findings
    return "PASS", findings


def _transform_findings(
    transform: TransformSummary, config: RegistrationConfig
) -> list[RegistrationFinding]:
    findings: list[RegistrationFinding] = []
    if any(not np.isfinite(item) for item in transform.parameters):
        findings.append(
            _finding(
                "REG-QC-CONV-001",
                "CRITICAL",
                "Transform has non-finite parameters.",
            )
        )
    translation = math.sqrt(sum(item * item for item in transform.translation_mm_xyz))
    if translation > config.maximum_translation_mm:
        findings.append(
            _finding(
                "REG-QC-TRN-001",
                "ERROR",
                "Transform translation exceeds configured maximum.",
                translation,
                config.maximum_translation_mm,
            )
        )
    rotation = max(abs(item) for item in transform.rotation_degrees)
    if rotation > config.maximum_rotation_degrees:
        findings.append(
            _finding(
                "REG-QC-ROT-001",
                "ERROR",
                "Transform rotation exceeds configured maximum.",
                rotation,
                config.maximum_rotation_degrees,
            )
        )
    if transform.affine_scale is not None:
        lower, upper = config.affine_scale_bounds
        if any(item < lower or item > upper for item in transform.affine_scale):
            findings.append(_finding("REG-QC-AFF-001", "ERROR", "Affine scale is implausible."))
    if transform.affine_shear is not None and transform.affine_shear > config.maximum_shear:
        findings.append(_finding("REG-QC-AFF-001", "ERROR", "Affine shear is implausible."))
    return findings


def _metric_findings(
    before: RegistrationMetrics, after: RegistrationMetrics, config: RegistrationConfig
) -> list[RegistrationFinding]:
    findings: list[RegistrationFinding] = []
    improvement = before.mean_squared_error - after.mean_squared_error
    if improvement < config.minimum_metric_improvement:
        findings.append(
            _finding(
                "REG-QC-MET-001",
                "ERROR",
                "Registration did not achieve configured MSE improvement.",
                improvement,
                config.minimum_metric_improvement,
            )
        )
    if (
        before.centre_of_mass_distance_mm is not None
        and after.centre_of_mass_distance_mm is not None
        and after.centre_of_mass_distance_mm > before.centre_of_mass_distance_mm
    ):
        findings.append(
            _finding(
                "REG-QC-MET-001",
                "ERROR",
                "Centre-of-mass distance increased after registration.",
                after.centre_of_mass_distance_mm,
                before.centre_of_mass_distance_mm,
            )
        )
    return findings


def _padding_findings(
    registered: np.ndarray,
    default_pixel_value: float,
    config: RegistrationConfig,
) -> list[RegistrationFinding]:
    padding_fraction = float(np.count_nonzero(registered == default_pixel_value) / registered.size)
    if padding_fraction > config.maximum_padding_fraction:
        return [
            _finding(
                "REG-QC-PAD-001",
                "ERROR",
                "Registered volume is dominated by default padding values.",
                padding_fraction,
                config.maximum_padding_fraction,
            )
        ]
    return []


def _finding(
    rule_id: str,
    severity: str,
    message: str,
    observed_value: object | None = None,
    expected_value: object | None = None,
) -> RegistrationFinding:
    return RegistrationFinding(
        rule_id=rule_id,
        severity=severity,  # type: ignore[arg-type]
        status="FAIL",
        message=message,
        observed_value=observed_value,
        expected_value=expected_value,
        remediation="Review registration configuration, inputs, and transform plausibility.",
    )
