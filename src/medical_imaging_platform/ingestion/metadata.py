"""Safe DICOM metadata extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydicom.dataset import FileDataset

from medical_imaging_platform.ingestion.models import DicomFileMetadata

IDENTIFYING_KEYWORDS = {
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "AccessionNumber",
    "StudyID",
    "OtherPatientIDs",
    "OtherPatientNames",
}


def extract_metadata(dataset: FileDataset, file_path: Path) -> DicomFileMetadata:
    """Extract safe technical metadata without direct identifier values."""
    return DicomFileMetadata(
        file_path=str(file_path),
        study_instance_uid=_str_or_none(dataset, "StudyInstanceUID"),
        series_instance_uid=_str_or_none(dataset, "SeriesInstanceUID"),
        sop_instance_uid=_str_or_none(dataset, "SOPInstanceUID"),
        modality=_str_or_none(dataset, "Modality"),
        body_part_examined=_str_or_none(dataset, "BodyPartExamined"),
        study_date=_str_or_none(dataset, "StudyDate"),
        series_description=_str_or_none(dataset, "SeriesDescription"),
        manufacturer=_str_or_none(dataset, "Manufacturer"),
        manufacturer_model_name=_str_or_none(dataset, "ManufacturerModelName"),
        rows=_int_or_none(dataset, "Rows"),
        columns=_int_or_none(dataset, "Columns"),
        pixel_spacing=_float_tuple(dataset, "PixelSpacing", 2),
        slice_thickness=_float_or_none(dataset, "SliceThickness"),
        image_orientation_patient=_float_tuple(dataset, "ImageOrientationPatient", 6),
        image_position_patient=_float_tuple(dataset, "ImagePositionPatient", 3),
        slice_location=_float_or_none(dataset, "SliceLocation"),
        instance_number=_int_or_none(dataset, "InstanceNumber"),
        rescale_slope=_float_or_none(dataset, "RescaleSlope"),
        rescale_intercept=_float_or_none(dataset, "RescaleIntercept"),
        photometric_interpretation=_str_or_none(dataset, "PhotometricInterpretation"),
        bits_allocated=_int_or_none(dataset, "BitsAllocated"),
        bits_stored=_int_or_none(dataset, "BitsStored"),
        pixel_representation=_int_or_none(dataset, "PixelRepresentation"),
        transfer_syntax_uid=str(dataset.file_meta.TransferSyntaxUID)
        if "TransferSyntaxUID" in dataset.file_meta
        else None,
        burned_in_annotation=_str_or_none(dataset, "BurnedInAnnotation"),
        has_pixel_data="PixelData" in dataset,
        private_tag_count=sum(1 for element in dataset.iterall() if element.tag.is_private),
    )


def detect_identifying_keywords(dataset: FileDataset) -> list[str]:
    """Return identifying DICOM keywords present in a dataset without values."""
    return sorted(keyword for keyword in IDENTIFYING_KEYWORDS if keyword in dataset)


def _str_or_none(dataset: FileDataset, keyword: str) -> str | None:
    value = getattr(dataset, keyword, None)
    if value is None:
        return None
    return str(value)


def _int_or_none(dataset: FileDataset, keyword: str) -> int | None:
    value = getattr(dataset, keyword, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(dataset: FileDataset, keyword: str) -> float | None:
    value = getattr(dataset, keyword, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_tuple(dataset: FileDataset, keyword: str, length: int) -> Any | None:
    value = getattr(dataset, keyword, None)
    if value is None or len(value) != length:
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
