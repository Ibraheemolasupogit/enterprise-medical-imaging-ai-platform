"""Typed models for deterministic CT preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CropMode = Literal["none", "non_background", "centre", "fixed"]
NormalisationMode = Literal["none", "minmax", "zscore"]
OutputFormat = Literal["npy", "npz"]
PixelDType = Literal["float32", "float64"]


class IntensityProfile(BaseModel):
    """Named clipping/windowing profile."""

    model_config = ConfigDict(extra="forbid")

    lower: float | None = None
    upper: float | None = None

    @model_validator(mode="after")
    def bounds_must_be_ordered(self) -> IntensityProfile:
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError("intensity profile lower bound must be less than upper bound")
        return self


class FixedCropBounds(BaseModel):
    """Half-open crop bounds in canonical [z, y, x] order."""

    model_config = ConfigDict(extra="forbid")

    z: tuple[int, int]
    y: tuple[int, int]
    x: tuple[int, int]

    @field_validator("z", "y", "x")
    @classmethod
    def bounds_are_ordered(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value[0] < 0 or value[1] <= value[0]:
            raise ValueError("crop bounds must be non-negative half-open intervals")
        return value


class PreprocessingConfig(BaseModel):
    """Configurable Milestone 5 preprocessing policy."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1)
    axis_order: Literal["zyx"] = "zyx"
    require_quality_report: bool = True
    allow_quality_override: bool = False
    pixel_dtype: PixelDType = "float32"
    missing_rescale_slope_default: float = 1.0
    missing_rescale_intercept_default: float = 0.0
    intensity_profiles: dict[str, IntensityProfile] = Field(min_length=1)
    default_intensity_profile: str
    normalisation_mode: NormalisationMode = "none"
    crop_mode: CropMode = "none"
    centre_crop_shape: tuple[int, int, int] | None = None
    fixed_crop_bounds: FixedCropBounds | None = None
    minimum_output_shape: tuple[int, int, int] | None = None
    padding_value: float = 0.0
    output_format: OutputFormat = "npy"
    output_directory: Path = Path("data/processed/preprocessing")
    overwrite: bool = False
    background_threshold: float = 0.0
    spacing_irregularity_tolerance_mm: float = 0.01

    @field_validator("centre_crop_shape", "minimum_output_shape")
    @classmethod
    def shape_values_must_be_positive(
        cls, value: tuple[int, int, int] | None
    ) -> tuple[int, int, int] | None:
        if value is not None and any(item <= 0 for item in value):
            raise ValueError("shape values must be positive")
        return value

    @model_validator(mode="after")
    def configured_references_must_exist(self) -> PreprocessingConfig:
        if self.default_intensity_profile not in self.intensity_profiles:
            raise ValueError("default_intensity_profile must name a configured profile")
        if self.crop_mode == "centre" and self.centre_crop_shape is None:
            raise ValueError("centre crop mode requires centre_crop_shape")
        if self.crop_mode == "fixed" and self.fixed_crop_bounds is None:
            raise ValueError("fixed crop mode requires fixed_crop_bounds")
        return self


class PixelConversionSummary(BaseModel):
    """Trace of raw-to-rescaled intensity conversion."""

    model_config = ConfigDict(extra="forbid")

    output_dtype: str
    terminology: str
    raw_range: tuple[float, float]
    converted_range: tuple[float, float]
    slope_values: list[float]
    intercept_values: list[float]
    defaulted_slope_sop_uids: list[str]
    defaulted_intercept_sop_uids: list[str]


class GeometrySummary(BaseModel):
    """Internal geometry metadata preserved for future milestones."""

    model_config = ConfigDict(extra="forbid")

    spacing_mm: tuple[float, float, float]
    spacing_source: str
    slice_spacing_median_mm: float | None
    slice_spacing_min_mm: float | None
    slice_spacing_max_mm: float | None
    slice_spacing_irregular: bool
    slice_spacing_reliable: bool
    slice_thickness_fallback_used: bool
    row_direction_cosines: tuple[float, float, float] | None
    column_direction_cosines: tuple[float, float, float] | None
    slice_normal: tuple[float, float, float] | None
    image_positions_patient: list[tuple[float, float, float] | None]
    orientation_classification: Literal["axial_like", "oblique", "indeterminate"]
    geometry_note: str


class IntensityTransformSummary(BaseModel):
    """Trace of intensity clipping/windowing/normalisation."""

    model_config = ConfigDict(extra="forbid")

    profile_name: str
    clipping_lower: float | None
    clipping_upper: float | None
    normalisation_mode: NormalisationMode
    input_range: tuple[float, float]
    clipped_range: tuple[float, float]
    output_range: tuple[float, float]
    clipped_voxel_count: int
    mean_before_normalisation: float
    std_before_normalisation: float
    fallback_used: bool
    output_dtype: str


class CropPaddingSummary(BaseModel):
    """Trace of deterministic crop and padding operations."""

    model_config = ConfigDict(extra="forbid")

    crop_mode: CropMode
    input_shape: tuple[int, int, int]
    crop_bounds_zyx: dict[str, tuple[int, int]]
    crop_offsets_zyx: tuple[int, int, int]
    cropped_shape: tuple[int, int, int]
    padding_value: float
    pad_widths_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    output_shape: tuple[int, int, int]


class PreprocessingOutputPaths(BaseModel):
    """Paths written by one preprocessing run."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str
    volume: str
    metadata: str
    report: str
    optional_npz: str | None = None


class PreprocessingResult(BaseModel):
    """Deterministic metadata report for one preprocessed volume."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    study_instance_uid: str
    series_instance_uid: str
    source_quality_report_id: str | None
    source_quality_status: str | None
    volume_shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    axis_order: Literal["z,y,x"]
    orientation_classification: str
    geometry: GeometrySummary
    pixel_conversion: PixelConversionSummary
    intensity_transform: IntensityTransformSummary
    crop_padding: CropPaddingSummary
    slice_count_input: int
    slice_count_output: int
    sop_instance_uid_to_index: dict[str, int]
    unreadable_files: list[str]
    excluded_files: list[str]
    output_paths: PreprocessingOutputPaths
    checksums: dict[str, str]
    policy_version: str
    generated_at: str
    warnings: list[str]
    override_used: bool
    override_reason: str | None
    resampling_deferred: bool = True
