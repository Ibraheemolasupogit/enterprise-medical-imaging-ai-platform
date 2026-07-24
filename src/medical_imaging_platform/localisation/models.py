"""Typed models for adrenal-region placeholder localisation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Side = Literal["left", "right"]
LocalisationMode = Literal["atlas"]
LocalisationStatus = Literal["LOCALISED", "LOCALISED_WITH_WARNINGS", "FAILED", "REJECTED"]
Severity = Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
FindingStatus = Literal["PASS", "FAIL", "NOT_EVALUATED"]


class LocalisationConfig(BaseModel):
    """Milestone 7 atlas-style localisation policy."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1)
    default_mode: LocalisationMode = "atlas"
    axis_order: Literal["z,y,x"] = "z,y,x"
    left_relative_centre: tuple[float, float, float]
    right_relative_centre: tuple[float, float, float]
    left_physical_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_physical_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    roi_size_voxels: tuple[int, int, int] | None = (8, 8, 8)
    roi_size_mm: tuple[float, float, float] | None = None
    padding_value: float = 0.0
    maximum_padding_fraction: float = Field(default=0.4, ge=0, le=1)
    maximum_centre_distance_mm: float = Field(default=5.0, gt=0)
    minimum_target_coverage: float = Field(default=0.8, ge=0, le=1)
    minimum_box_iou: float = Field(default=0.2, ge=0, le=1)
    left_right_minimum_separation_mm: float = Field(default=4.0, gt=0)
    confidence_weights: dict[str, float] = Field(default_factory=dict)
    quality_status_policy: dict[str, str] = Field(default_factory=dict)
    output_directory: Path = Path("data/processed/localisation")
    overwrite: bool = False

    @field_validator("left_relative_centre", "right_relative_centre")
    @classmethod
    def relative_coordinates_bounded(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if any(item < 0 or item > 1 for item in value):
            raise ValueError("relative centre coordinates must be within [0, 1]")
        return value

    @field_validator("roi_size_voxels")
    @classmethod
    def voxel_roi_positive(cls, value: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
        if value is not None and any(item <= 0 for item in value):
            raise ValueError("roi_size_voxels values must be positive")
        return value

    @field_validator("roi_size_mm")
    @classmethod
    def physical_roi_positive(
        cls, value: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if value is not None and any(item <= 0 for item in value):
            raise ValueError("roi_size_mm values must be positive")
        return value

    @model_validator(mode="after")
    def definitions_are_valid(self) -> LocalisationConfig:
        if self.left_relative_centre == self.right_relative_centre:
            raise ValueError("left and right relative centres must be distinct")
        if (self.roi_size_voxels is None) == (self.roi_size_mm is None):
            raise ValueError("configure exactly one of roi_size_voxels or roi_size_mm")
        return self


class SourceVolumeMetadata(BaseModel):
    """Validated localisation input metadata."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["preprocessing", "registration"]
    source_dir: str
    source_run_id: str
    source_status: str | None
    upstream_override_used: bool
    volume_shape: tuple[int, int, int]
    spacing_mm_zyx: tuple[float, float, float]
    axis_order: str


class BoundingBox(BaseModel):
    """Half-open [z, y, x] box."""

    model_config = ConfigDict(extra="forbid")

    z: tuple[int, int]
    y: tuple[int, int]
    x: tuple[int, int]


class RoiExtraction(BaseModel):
    """ROI extraction trace."""

    model_config = ConfigDict(extra="forbid")

    roi_shape: tuple[int, int, int]
    crop_bounds_zyx: BoundingBox
    crop_offsets_zyx: tuple[int, int, int]
    pad_widths_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    padding_fraction: float
    source_spacing_mm_zyx: tuple[float, float, float]


class SideLocalisation(BaseModel):
    """One side localisation output."""

    model_config = ConfigDict(extra="forbid")

    side: Side
    predicted_centre_voxel: tuple[int, int, int]
    predicted_centre_mm: tuple[float, float, float]
    bounding_box_voxel: BoundingBox
    bounding_box_mm: dict[str, tuple[float, float]]
    roi_shape: tuple[int, int, int]
    confidence: float
    status: LocalisationStatus
    warnings: list[str]
    roi_extraction: RoiExtraction


class LocalisationMetrics(BaseModel):
    """Synthetic technical localisation metrics."""

    model_config = ConfigDict(extra="forbid")

    centre_distance_voxels: float | None
    centre_distance_mm: float | None
    bounding_box_iou: float | None
    target_coverage: float | None
    predicted_roi_volume_voxels: int
    ground_truth_volume_voxels: int | None
    left_right_consistency: bool
    missed_target: bool | None
    left_right_swap: bool | None
    evaluation_status: Literal["AVAILABLE", "NOT_EVALUATED"]


class LocalisationFinding(BaseModel):
    """One localisation quality finding."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    severity: Severity
    status: FindingStatus
    message: str
    side: Side | None = None
    observed_value: object | None = None
    expected_value: object | None = None
    remediation: str


class LocalisationOutputPaths(BaseModel):
    """Output paths for one localisation run."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str
    left_roi: str
    right_roi: str
    localisation_json: str
    report: str
    left_overlay: str
    right_overlay: str


class LocalisationResult(BaseModel):
    """Full localisation report."""

    model_config = ConfigDict(extra="forbid")

    localisation_run_id: str
    source: SourceVolumeMetadata
    localisation_mode: LocalisationMode
    configuration_version: str
    left: SideLocalisation
    right: SideLocalisation
    ground_truth_available: bool
    metrics: dict[Side, LocalisationMetrics]
    quality_findings: list[LocalisationFinding]
    overall_status: LocalisationStatus
    warnings: list[str]
    output_paths: LocalisationOutputPaths
    checksums: dict[str, str]
    generated_at: str
    limitations: list[str]
    recommended_next_action: str
