"""Typed models for longitudinal synthetic lesion analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LongitudinalLabel = Literal["new", "increased", "stable", "reduced", "resolved", "indeterminate"]
LongitudinalStatus = Literal["PASS", "PASS_WITH_WARNINGS", "FAIL", "REJECTED"]


class LongitudinalConfig(BaseModel):
    """Milestone 10 longitudinal analysis policy."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1)
    minimum_component_voxels: int = Field(ge=1)
    maximum_match_distance_mm: float = Field(gt=0)
    minimum_match_iou: float = Field(ge=0, le=1)
    minimum_match_dice: float = Field(ge=0, le=1)
    centroid_weight: float = Field(ge=0)
    overlap_weight: float = Field(ge=0)
    volume_increase_threshold_percent: float = Field(ge=0)
    volume_reduction_threshold_percent: float = Field(ge=0)
    diameter_increase_threshold_percent: float = Field(ge=0)
    diameter_reduction_threshold_percent: float = Field(ge=0)
    small_denominator_mm3: float = Field(gt=0)
    require_registration_pass: bool = True
    require_segmentation_pass: bool = True
    classification_abstention_forces_indeterminate: bool = True
    allow_multiple_components: bool = True
    output_directory: Path = Path("ml/experiments/longitudinal")
    overwrite: bool = False

    @model_validator(mode="after")
    def weights_are_usable(self) -> LongitudinalConfig:
        if self.centroid_weight == 0 and self.overlap_weight == 0:
            raise ValueError("At least one matching weight must be positive.")
        return self


class PairManifest(BaseModel):
    """Input pair metadata recorded with longitudinal evidence."""

    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    case_id: str
    research_subject_id: str
    side: Literal["left", "right"]
    previous_timepoint: str
    current_timepoint: str
    previous_mask: str
    current_mask: str
    previous_spacing_mm: tuple[float, float, float]
    current_spacing_mm: tuple[float, float, float]
    registration_run_id: str | None = None
    localisation_run_ids: list[str] = Field(default_factory=list)
    segmentation_run_ids: list[str] = Field(default_factory=list)
    classification_run_ids: list[str] = Field(default_factory=list)
    source_checksums: dict[str, str]
    upstream_quality_statuses: dict[str, str]
    generated_at: str

    @field_validator("previous_spacing_mm", "current_spacing_mm")
    @classmethod
    def spacing_valid(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if len(value) != 3 or any(axis <= 0 for axis in value):
            raise ValueError("Spacing must contain three positive values.")
        return value


class LesionMeasurement(BaseModel):
    """Spacing-aware lesion measurement for one component or timepoint."""

    model_config = ConfigDict(extra="forbid")

    lesion_id: str
    timepoint: Literal["previous", "current"]
    component_index: int
    voxel_count: int
    physical_volume_mm3: float
    physical_volume_ml: float
    bounding_box_voxel: list[list[int]]
    bounding_box_dimensions_mm: tuple[float, float, float]
    maximum_3d_diameter_mm: float
    axial_maximum_diameter_mm: float
    centroid_voxel: tuple[float, float, float] | None
    centroid_physical_mm: tuple[float, float, float] | None
    connected_component_count: int
    empty_mask: bool


class LesionMatch(BaseModel):
    """One deterministic previous/current component match candidate or result."""

    model_config = ConfigDict(extra="forbid")

    match_id: str
    previous_lesion_id: str | None
    current_lesion_id: str | None
    centroid_distance_mm: float | None
    bounding_box_iou: float | None
    overlap_dice: float | None
    overlap_iou: float | None
    matching_score: float
    status: Literal["matched", "unmatched_previous", "unmatched_current", "ambiguous"]
    ambiguous: bool = False


class LongitudinalChange(BaseModel):
    """Calculated change and engineering label for one match outcome."""

    model_config = ConfigDict(extra="forbid")

    change_id: str
    label: LongitudinalLabel
    previous_lesion_id: str | None
    current_lesion_id: str | None
    absolute_volume_change_mm3: float | None
    percentage_volume_change: float | None
    absolute_diameter_change_mm: float | None
    percentage_diameter_change: float | None
    centroid_displacement_mm: float | None
    overlap_dice: float | None
    overlap_iou: float | None
    reasons: list[str]


class LongitudinalFinding(BaseModel):
    """One longitudinal quality finding."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    severity: Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
    status: Literal["PASS", "FAIL", "NOT_EVALUATED"]
    message: str
    observed_value: object | None = None
    expected_value: object | None = None
    remediation: str
