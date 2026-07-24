"""Typed models for DICOM ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Severity = Literal["INFO", "WARNING", "ERROR"]
OrderingStrategy = Literal[
    "image_position_patient", "slice_location", "instance_number", "filename"
]


class DicomIngestionConfig(BaseModel):
    """Typed DICOM ingestion and de-identification configuration."""

    model_config = ConfigDict(extra="forbid")

    input_dir: Path
    output_dir: Path
    audit_output_dir: Path
    fixture_output_dir: Path
    max_file_size_bytes: int = Field(gt=0)
    max_files: int = Field(gt=0)
    accepted_modality: str = Field(min_length=1)
    required_metadata: list[str] = Field(min_length=1)
    ordering_fallback_policy: list[OrderingStrategy] = Field(min_length=1)
    private_tag_policy: dict[str, object]
    deidentification: dict[str, object]

    @field_validator("accepted_modality")
    @classmethod
    def modality_must_be_uppercase(cls, value: str) -> str:
        return value.upper()


class DicomFileMetadata(BaseModel):
    """Safe technical DICOM metadata for normal application output."""

    model_config = ConfigDict(extra="forbid")

    file_path: str
    study_instance_uid: str | None
    series_instance_uid: str | None
    sop_instance_uid: str | None
    modality: str | None
    body_part_examined: str | None = None
    study_date: str | None = None
    series_description: str | None = None
    manufacturer: str | None = None
    manufacturer_model_name: str | None = None
    rows: int | None = None
    columns: int | None = None
    pixel_spacing: tuple[float, float] | None = None
    slice_thickness: float | None = None
    image_orientation_patient: tuple[float, float, float, float, float, float] | None = None
    image_position_patient: tuple[float, float, float] | None = None
    slice_location: float | None = None
    instance_number: int | None = None
    rescale_slope: float | None = None
    rescale_intercept: float | None = None
    photometric_interpretation: str | None = None
    bits_allocated: int | None = None
    bits_stored: int | None = None
    pixel_representation: int | None = None
    transfer_syntax_uid: str | None = None
    burned_in_annotation: str | None = None
    has_pixel_data: bool = False
    private_tag_count: int = 0


class SkippedDicomFile(BaseModel):
    """A skipped file and its reason."""

    path: str
    reason: str


class DicomSeries(BaseModel):
    """Discovered DICOM series."""

    study_instance_uid: str
    series_instance_uid: str
    files: list[str]


class DicomDiscoveryResult(BaseModel):
    """Recursive discovery result."""

    series: list[DicomSeries]
    skipped_files: list[SkippedDicomFile]


class OrderingIssue(BaseModel):
    """Slice ordering issue."""

    severity: Severity
    message: str


class OrderedSliceSet(BaseModel):
    """Ordered slice metadata and strategy."""

    strategy: OrderingStrategy
    files: list[DicomFileMetadata]
    issues: list[OrderingIssue]


class ValidationFinding(BaseModel):
    """Structured DICOM validation finding."""

    rule_id: str
    severity: Severity
    message: str
    file_path: str | None = None
    study_uid: str | None = None
    series_uid: str | None = None
