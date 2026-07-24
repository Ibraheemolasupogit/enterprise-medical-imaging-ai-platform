"""DICOM metadata quality checks."""

from __future__ import annotations

from medical_imaging_platform.ingestion.models import DicomFileMetadata
from medical_imaging_platform.quality_control.models import QualityControlConfig, QualityFinding
from medical_imaging_platform.quality_control.rules import rule_info


def run_metadata_checks(
    metadata: list[DicomFileMetadata],
    config: QualityControlConfig,
) -> tuple[list[QualityFinding], dict[str, object]]:
    """Run metadata consistency and security checks."""
    findings: list[QualityFinding] = []
    files = [item.file_path for item in metadata]
    _check_values(findings, metadata, "study_instance_uid", "DICOM-QC-UID-001", True)
    _check_values(findings, metadata, "series_instance_uid", "DICOM-QC-UID-001", True)
    _check_values(findings, metadata, "sop_instance_uid", "DICOM-QC-SOP-001", True, unique=True)
    _check_values(findings, metadata, "rows", "DICOM-QC-DIM-001", True)
    _check_values(findings, metadata, "columns", "DICOM-QC-DIM-001", True)
    _check_values(findings, metadata, "pixel_spacing", "DICOM-QC-SPC-001", True)
    _check_values(findings, metadata, "pixel_spacing", "DICOM-QC-SPC-002", False)
    _check_values(findings, metadata, "slice_thickness", "DICOM-QC-THK-001", True)
    _check_values(findings, metadata, "slice_thickness", "DICOM-QC-THK-002", False)
    _check_values(findings, metadata, "image_orientation_patient", "DICOM-QC-ORI-001", True)
    _check_values(findings, metadata, "image_orientation_patient", "DICOM-QC-ORI-002", False)

    modalities = {item.modality for item in metadata}
    if modalities != set(config.accepted_modalities):
        _append(
            findings,
            "DICOM-QC-MOD-001",
            "FAIL",
            "Series modality does not match accepted modality configuration.",
            files,
            sorted(str(item) for item in modalities),
            config.accepted_modalities,
            "Use CT-only input for this milestone.",
        )

    body_parts = {
        str(item.body_part_examined).upper() for item in metadata if item.body_part_examined
    }
    allowed = {item.upper() for item in config.body_region_allowlist}
    if config.require_body_part_examined and not body_parts:
        _append(
            findings,
            "DICOM-QC-BODY-001",
            "FAIL",
            "BodyPartExamined is missing; anatomy is not inferred from pixels.",
            files,
            None,
            sorted(allowed),
            "Review source metadata.",
        )
    elif body_parts and not body_parts.issubset(allowed):
        _append(
            findings,
            "DICOM-QC-BODY-001",
            "FAIL",
            "Body-region metadata is outside the configured allowlist.",
            files,
            sorted(body_parts),
            sorted(allowed),
            "Review body-region metadata and configuration.",
        )

    transfer_syntaxes = {item.transfer_syntax_uid for item in metadata}
    unsupported = transfer_syntaxes - set(config.supported_transfer_syntaxes)
    if unsupported:
        _append(
            findings,
            "DICOM-QC-TSX-001",
            "FAIL",
            "Unsupported transfer syntax detected.",
            files,
            sorted(str(item) for item in unsupported),
            config.supported_transfer_syntaxes,
            "Use supported uncompressed fixtures or add codec support in a future milestone.",
        )

    burned_values = {str(item.burned_in_annotation).upper() for item in metadata}
    if "YES" in burned_values:
        _append(
            findings,
            "DICOM-QC-PHI-001",
            "FAIL",
            "BurnedInAnnotation is YES; metadata cannot remove pixel PHI.",
            files,
            sorted(burned_values),
            "NO",
            "Quarantine or manually review the series.",
        )
    elif "NONE" in burned_values or "UNKNOWN" in burned_values:
        _append(
            findings,
            "DICOM-QC-PHI-001",
            "FAIL",
            "BurnedInAnnotation is missing or unknown.",
            files,
            sorted(burned_values),
            "NO",
            "Manual review is required before using public DICOM data.",
        )

    private_count = sum(item.private_tag_count for item in metadata)
    if private_count:
        _append(
            findings,
            "DICOM-QC-PRV-001",
            "FAIL",
            "Private tags are present.",
            files,
            private_count,
            0,
            "Run metadata de-identification before downstream use.",
        )

    metrics: dict[str, object] = {
        "modalities": sorted(str(item) for item in modalities),
        "body_parts": sorted(body_parts),
        "private_tag_count": private_count,
        "burned_in_annotation_values": sorted(burned_values),
    }
    return findings, metrics


def _check_values(
    findings: list[QualityFinding],
    metadata: list[DicomFileMetadata],
    field_name: str,
    rule_id: str,
    required: bool,
    unique: bool = False,
) -> None:
    values = [getattr(item, field_name) for item in metadata]
    files = [item.file_path for item in metadata]
    if required and any(value is None for value in values):
        _append(
            findings,
            rule_id,
            "FAIL",
            f"{field_name} is missing.",
            files,
            None,
            "present",
            "Review source metadata.",
        )
        return
    present = [value for value in values if value is not None]
    comparable = [_normalise(value) for value in present]
    if unique:
        if len(set(comparable)) != len(comparable):
            _append(
                findings,
                rule_id,
                "FAIL",
                f"{field_name} values are not unique.",
                files,
                "duplicate",
                "unique",
                "Review duplicate identifiers.",
            )
    elif present and len(set(comparable)) > 1:
        _append(
            findings,
            rule_id,
            "FAIL",
            f"{field_name} is inconsistent.",
            files,
            sorted(str(item) for item in set(comparable)),
            "consistent",
            "Review mixed metadata.",
        )


def _normalise(value: object) -> object:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, tuple):
        return tuple(
            round(float(item), 4) if isinstance(item, float | int) else item for item in value
        )
    return value


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
