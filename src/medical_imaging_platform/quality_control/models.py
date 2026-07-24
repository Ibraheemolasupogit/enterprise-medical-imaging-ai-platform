"""Typed quality-control models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Severity = Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
RuleStatus = Literal["PASS", "FAIL", "NOT_APPLICABLE", "NOT_EVALUATED"]
SeriesQualityStatus = Literal["PASS", "PASS_WITH_WARNINGS", "FAIL", "REJECTED"]
FindingCategory = Literal["FILE", "METADATA", "SLICE", "PIXEL", "SECURITY", "SCORING"]


class QualityControlConfig(BaseModel):
    """Typed Milestone 4 DICOM quality-control configuration."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1)
    output_dir: Path
    accepted_modalities: list[str] = Field(min_length=1)
    body_region_allowlist: list[str] = Field(min_length=1)
    require_body_part_examined: bool
    pixel_spacing_tolerance: float = Field(ge=0)
    slice_thickness_tolerance: float = Field(ge=0)
    orientation_tolerance: float = Field(ge=0)
    slice_gap_tolerance: float = Field(ge=0)
    minimum_slice_count: int = Field(ge=1)
    maximum_slice_count: int = Field(ge=1)
    maximum_corrupt_file_ratio: float = Field(ge=0, le=1)
    burned_in_annotation_policy: str = Field(min_length=1)
    private_tag_policy: str = Field(min_length=1)
    supported_transfer_syntaxes: list[str] = Field(min_length=1)
    pixel_value_bounds: tuple[float, float]
    scoring_weights: dict[str, int]
    status_thresholds: dict[str, int]
    critical_rule_ids: list[str]

    @field_validator("accepted_modalities")
    @classmethod
    def modalities_uppercase(cls, value: list[str]) -> list[str]:
        return [item.upper() for item in value]

    @field_validator("pixel_value_bounds")
    @classmethod
    def pixel_bounds_ordered(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] >= value[1]:
            raise ValueError("pixel_value_bounds lower bound must be less than upper bound")
        return value


class QualityFinding(BaseModel):
    """One quality-control finding."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    category: FindingCategory
    severity: Severity
    status: RuleStatus
    message: str
    affected_files: list[str] = Field(default_factory=list)
    observed_value: Any | None = None
    expected_value: Any | None = None
    remediation: str


class QualityReport(BaseModel):
    """Series-level DICOM quality report."""

    model_config = ConfigDict(extra="forbid")

    quality_report_id: str
    study_instance_uid: str | None
    series_instance_uid: str | None
    status: SeriesQualityStatus
    quality_score: int
    evaluated_file_count: int
    expected_slice_count: int | None
    observed_slice_count: int
    ordering_strategy: str
    findings: list[QualityFinding]
    metrics: dict[str, Any]
    generated_at: str
    policy_version: str
    score_deductions: dict[str, int]
