"""Stable DICOM quality-control rule catalogue."""

from __future__ import annotations

from medical_imaging_platform.quality_control.models import FindingCategory, Severity

RULE_CATALOGUE: dict[str, tuple[FindingCategory, Severity, str]] = {
    "DICOM-QC-MOD-001": ("METADATA", "ERROR", "Accepted modality"),
    "DICOM-QC-BODY-001": ("METADATA", "WARNING", "Expected body-region metadata"),
    "DICOM-QC-UID-001": ("METADATA", "ERROR", "Study and series UID consistency"),
    "DICOM-QC-SOP-001": ("METADATA", "ERROR", "Unique SOP Instance UIDs"),
    "DICOM-QC-DIM-001": ("METADATA", "ERROR", "Consistent rows and columns"),
    "DICOM-QC-SPC-001": ("METADATA", "ERROR", "Pixel-spacing presence"),
    "DICOM-QC-SPC-002": ("METADATA", "ERROR", "Pixel-spacing consistency"),
    "DICOM-QC-THK-001": ("METADATA", "WARNING", "Slice-thickness presence"),
    "DICOM-QC-THK-002": ("METADATA", "WARNING", "Slice-thickness consistency"),
    "DICOM-QC-ORI-001": ("METADATA", "ERROR", "Orientation presence"),
    "DICOM-QC-ORI-002": ("METADATA", "ERROR", "Orientation consistency"),
    "DICOM-QC-ORD-001": ("SLICE", "ERROR", "Deterministic slice ordering"),
    "DICOM-QC-SLC-001": ("SLICE", "ERROR", "Duplicate slice positions"),
    "DICOM-QC-SLC-002": ("SLICE", "WARNING", "Missing or irregular slice gaps"),
    "DICOM-QC-SLC-003": ("SLICE", "ERROR", "Duplicate instance numbers"),
    "DICOM-QC-PIX-001": ("PIXEL", "ERROR", "Pixel data presence"),
    "DICOM-QC-PIX-002": ("PIXEL", "ERROR", "Pixel array readable"),
    "DICOM-QC-PIX-003": ("PIXEL", "ERROR", "Pixel dimensions match metadata"),
    "DICOM-QC-PIX-004": ("PIXEL", "WARNING", "Pixel value range technically plausible"),
    "DICOM-QC-TSX-001": ("FILE", "ERROR", "Supported transfer syntax"),
    "DICOM-QC-PHI-001": ("SECURITY", "CRITICAL", "Burned-in annotation status"),
    "DICOM-QC-PRV-001": ("SECURITY", "WARNING", "Private-tag presence"),
    "DICOM-QC-FILE-001": ("FILE", "ERROR", "Corrupt or unreadable files"),
}


def rule_info(rule_id: str) -> tuple[FindingCategory, Severity, str]:
    """Return category, default severity, and title for a rule."""
    return RULE_CATALOGUE[rule_id]
