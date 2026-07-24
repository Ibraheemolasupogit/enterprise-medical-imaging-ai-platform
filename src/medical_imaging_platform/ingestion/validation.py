"""Basic structural DICOM series validation."""

from __future__ import annotations

from pathlib import Path

from pydicom.uid import ExplicitVRLittleEndian

from medical_imaging_platform.ingestion.loader import load_dicom
from medical_imaging_platform.ingestion.metadata import extract_metadata
from medical_imaging_platform.ingestion.models import DicomFileMetadata, ValidationFinding
from medical_imaging_platform.ingestion.ordering import order_slices


def validate_series(
    file_paths: list[Path],
    *,
    accepted_modality: str,
    max_file_size_bytes: int,
    require_pixel_data: bool = False,
) -> list[ValidationFinding]:
    """Run basic structural validation for one DICOM series."""
    findings: list[ValidationFinding] = []
    metadata: list[DicomFileMetadata] = []
    for path in sorted(file_paths):
        try:
            dataset = load_dicom(
                path, header_only=not require_pixel_data, max_file_size_bytes=max_file_size_bytes
            )
            item = extract_metadata(dataset, path)
            metadata.append(item)
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    rule_id="DICOM_READABLE",
                    severity="ERROR",
                    message=str(exc),
                    file_path=str(path),
                )
            )

    if not metadata:
        return findings

    study_uids = {item.study_instance_uid for item in metadata}
    series_uids = {item.series_instance_uid for item in metadata}
    if None in study_uids or None in series_uids:
        findings.append(
            _finding("UID_PRESENT", "ERROR", "Study and series UIDs are required", metadata[0])
        )
    if len(study_uids) > 1 or len(series_uids) > 1:
        findings.append(
            _finding("GROUPING_CONSISTENT", "ERROR", "Mixed study or series detected", metadata[0])
        )

    for item in metadata:
        if item.modality != accepted_modality:
            findings.append(
                _finding("MODALITY_CT", "ERROR", f"Expected modality {accepted_modality}", item)
            )
        if item.transfer_syntax_uid != str(ExplicitVRLittleEndian):
            findings.append(
                _finding("TRANSFER_SYNTAX", "ERROR", "Unsupported transfer syntax", item)
            )
        if require_pixel_data and not item.has_pixel_data:
            findings.append(
                _finding("PIXEL_DATA_PRESENT", "ERROR", "Pixel data are required", item)
            )

    sop_uids = [item.sop_instance_uid for item in metadata if item.sop_instance_uid]
    if len(set(sop_uids)) != len(sop_uids):
        findings.append(
            _finding("SOP_UID_UNIQUE", "ERROR", "Duplicate SOP Instance UID detected", metadata[0])
        )
    _consistent(metadata, "rows", "DIMENSIONS_CONSISTENT", findings)
    _consistent(metadata, "columns", "DIMENSIONS_CONSISTENT", findings)
    _consistent(metadata, "pixel_spacing", "PIXEL_SPACING_CONSISTENT", findings)
    _consistent(metadata, "slice_thickness", "SLICE_THICKNESS_CONSISTENT", findings)
    _consistent(metadata, "image_orientation_patient", "ORIENTATION_CONSISTENT", findings)

    ordering = order_slices(metadata)
    for issue in ordering.issues:
        findings.append(
            ValidationFinding(
                rule_id="SLICE_ORDERING",
                severity=issue.severity,
                message=issue.message,
                study_uid=metadata[0].study_instance_uid,
                series_uid=metadata[0].series_instance_uid,
            )
        )
    return findings


def _consistent(
    metadata: list[DicomFileMetadata],
    field_name: str,
    rule_id: str,
    findings: list[ValidationFinding],
) -> None:
    values = {getattr(item, field_name) for item in metadata}
    if None in values:
        findings.append(_finding(rule_id, "ERROR", f"{field_name} is required", metadata[0]))
    elif len(values) > 1:
        findings.append(_finding(rule_id, "ERROR", f"{field_name} is inconsistent", metadata[0]))


def _finding(
    rule_id: str, severity: str, message: str, item: DicomFileMetadata
) -> ValidationFinding:
    return ValidationFinding(
        rule_id=rule_id,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        file_path=item.file_path,
        study_uid=item.study_instance_uid,
        series_uid=item.series_instance_uid,
    )
