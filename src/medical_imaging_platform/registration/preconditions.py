"""Registration precondition checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from medical_imaging_platform.preprocessing.export import validate_preprocessed_volume
from medical_imaging_platform.preprocessing.models import PreprocessingResult
from medical_imaging_platform.registration.models import RegistrationFinding, VolumeRoleMetadata


def load_preprocessed_input(path: Path) -> tuple[np.ndarray, PreprocessingResult]:
    """Load and validate a preprocessing output directory."""
    metadata = validate_preprocessed_volume(path)
    volume = np.load(Path(metadata.output_paths.volume))
    return volume, metadata


def validate_registration_inputs(
    fixed_dir: Path,
    moving_dir: Path,
    *,
    allow_constant_volume: bool,
    fixed_temporal_label: str | None = None,
    moving_temporal_label: str | None = None,
) -> tuple[
    np.ndarray, PreprocessingResult, np.ndarray, PreprocessingResult, list[RegistrationFinding]
]:
    """Validate fixed and moving registration inputs before conversion."""
    fixed_volume, fixed_meta = load_preprocessed_input(fixed_dir)
    moving_volume, moving_meta = load_preprocessed_input(moving_dir)
    findings: list[RegistrationFinding] = []

    if fixed_meta.run_id == moving_meta.run_id:
        findings.append(
            _finding("REG-QC-INP-001", "CRITICAL", "Fixed and moving inputs are the same run.")
        )
    for role, volume, metadata in (
        ("fixed", fixed_volume, fixed_meta),
        ("moving", moving_volume, moving_meta),
    ):
        findings.extend(_volume_findings(role, volume, metadata, allow_constant_volume))
    if fixed_temporal_label is None or moving_temporal_label is None:
        findings.append(
            _finding(
                "REG-QC-INP-001",
                "WARNING",
                "Temporal labels were not provided; fixed/moving roles are explicit but "
                "temporal ordering is unspecified.",
            )
        )
    return fixed_volume, fixed_meta, moving_volume, moving_meta, findings


def role_metadata(
    role: str,
    preprocessing_dir: Path,
    metadata: PreprocessingResult,
    *,
    temporal_label: str | None,
) -> VolumeRoleMetadata:
    """Build persisted fixed/moving role metadata."""
    return VolumeRoleMetadata(
        role=role,  # type: ignore[arg-type]
        preprocessing_dir=str(preprocessing_dir),
        run_id=metadata.run_id,
        study_instance_uid=metadata.study_instance_uid,
        series_instance_uid=metadata.series_instance_uid,
        temporal_label=temporal_label,
        source_quality_report_id=metadata.source_quality_report_id,
        source_quality_status=metadata.source_quality_status,
        preprocessing_override_used=metadata.override_used,
        volume_shape=metadata.volume_shape,
        spacing_mm_zyx=metadata.spacing_mm,
        axis_order=metadata.axis_order,
    )


def has_rejection(findings: list[RegistrationFinding]) -> bool:
    return any(finding.severity == "CRITICAL" and finding.status == "FAIL" for finding in findings)


def _volume_findings(
    role: str,
    volume: np.ndarray,
    metadata: PreprocessingResult,
    allow_constant_volume: bool,
) -> list[RegistrationFinding]:
    findings: list[RegistrationFinding] = []
    if volume.ndim != 3:
        findings.append(_finding("REG-QC-INP-001", "CRITICAL", f"{role} volume is not 3D."))
    if not np.all(np.isfinite(volume)):
        findings.append(
            _finding("REG-QC-INP-001", "CRITICAL", f"{role} volume contains non-finite values.")
        )
    if volume.size == 0:
        findings.append(_finding("REG-QC-INP-001", "CRITICAL", f"{role} volume is empty."))
    if not allow_constant_volume and float(np.max(volume)) == float(np.min(volume)):
        findings.append(_finding("REG-QC-INP-001", "ERROR", f"{role} volume is constant."))
    if metadata.axis_order != "z,y,x":
        findings.append(_finding("REG-QC-GEO-001", "CRITICAL", f"{role} axis order is not z,y,x."))
    if any(item <= 0 or not np.isfinite(item) for item in metadata.spacing_mm):
        findings.append(_finding("REG-QC-GEO-001", "CRITICAL", f"{role} spacing is invalid."))
    if metadata.geometry is None:
        findings.append(
            _finding("REG-QC-GEO-001", "CRITICAL", f"{role} geometry metadata is missing.")
        )
    if metadata.source_quality_status == "REJECTED":
        findings.append(
            _finding("REG-QC-INP-001", "CRITICAL", f"{role} source QC status is REJECTED.")
        )
    if metadata.source_quality_status == "FAIL" and not metadata.override_used:
        findings.append(
            _finding(
                "REG-QC-INP-001",
                "ERROR",
                f"{role} source QC failure was not explicitly propagated.",
            )
        )
    if metadata.override_used:
        findings.append(
            _finding("REG-QC-INP-001", "WARNING", f"{role} preprocessing used an override.")
        )
    return findings


def _finding(rule_id: str, severity: str, message: str) -> RegistrationFinding:
    return RegistrationFinding(
        rule_id=rule_id,
        severity=severity,  # type: ignore[arg-type]
        status="FAIL",
        message=message,
        remediation="Review registration inputs and provenance before retrying.",
    )
