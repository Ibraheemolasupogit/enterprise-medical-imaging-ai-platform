"""Typed models for longitudinal registration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RegistrationMode = Literal["centre_of_mass", "rigid", "rigid_then_affine"]
InitialisationMethod = Literal["centre_of_mass", "geometry"]
RegistrationStatus = Literal["PASS", "PASS_WITH_WARNINGS", "FAIL", "REJECTED"]
Severity = Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
RuleOutcome = Literal["PASS", "FAIL"]


class RegistrationConfig(BaseModel):
    """Milestone 6 registration policy."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1)
    default_mode: RegistrationMode = "rigid"
    initialisation_method: InitialisationMethod = "centre_of_mass"
    allow_constant_volume: bool = False
    foreground_threshold: float = 0.0
    metric: Literal["mattes_mutual_information", "normalised_correlation"] = (
        "mattes_mutual_information"
    )
    metric_bins: int = Field(default=32, ge=2)
    metric_sampling_strategy: Literal["none", "regular", "random"] = "none"
    metric_sampling_percentage: float = Field(default=1.0, gt=0, le=1)
    random_seed: int = Field(default=20260724, ge=0)
    optimiser: Literal["gradient_descent"] = "gradient_descent"
    learning_rate: float = Field(default=1.0, gt=0)
    minimum_step: float = Field(default=0.001, gt=0)
    maximum_iterations: int = Field(default=50, ge=1)
    convergence_window_size: int = Field(default=5, ge=1)
    shrink_factors: list[int] = Field(default_factory=lambda: [2, 1], min_length=1)
    smoothing_sigmas: list[float] = Field(default_factory=lambda: [1.0, 0.0], min_length=1)
    interpolator: Literal["linear", "nearest"] = "linear"
    default_pixel_value: float = 0.0
    maximum_translation_mm: float = Field(default=100.0, gt=0)
    maximum_rotation_degrees: float = Field(default=30.0, gt=0)
    affine_scale_bounds: tuple[float, float] = (0.8, 1.25)
    maximum_shear: float = Field(default=0.25, ge=0)
    minimum_metric_improvement: float = 0.0
    maximum_padding_fraction: float = Field(default=0.5, ge=0, le=1)
    quality_status_policy: dict[str, str] = Field(default_factory=dict)
    output_directory: Path = Path("data/processed/registration")
    overwrite: bool = False

    @field_validator("shrink_factors")
    @classmethod
    def shrink_factors_positive(cls, value: list[int]) -> list[int]:
        if any(item <= 0 for item in value):
            raise ValueError("shrink_factors must be positive")
        return value

    @model_validator(mode="after")
    def lists_and_bounds_are_valid(self) -> RegistrationConfig:
        if len(self.shrink_factors) != len(self.smoothing_sigmas):
            raise ValueError("shrink_factors and smoothing_sigmas must have matching lengths")
        if (
            self.affine_scale_bounds[0] <= 0
            or self.affine_scale_bounds[0] >= self.affine_scale_bounds[1]
        ):
            raise ValueError("affine_scale_bounds must be positive and ordered")
        return self


class RegistrationFinding(BaseModel):
    """One registration quality or precondition finding."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    severity: Severity
    status: RuleOutcome
    message: str
    observed_value: object | None = None
    expected_value: object | None = None
    remediation: str


class VolumeRoleMetadata(BaseModel):
    """Fixed or moving source metadata."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["fixed", "moving"]
    preprocessing_dir: str
    run_id: str
    study_instance_uid: str
    series_instance_uid: str
    temporal_label: str | None = None
    source_quality_report_id: str | None = None
    source_quality_status: str | None = None
    preprocessing_override_used: bool
    volume_shape: tuple[int, int, int]
    spacing_mm_zyx: tuple[float, float, float]
    axis_order: str


class RegistrationMetrics(BaseModel):
    """Technical registration metrics."""

    model_config = ConfigDict(extra="forbid")

    mean_squared_error: float
    normalised_cross_correlation: float
    mutual_information: float
    foreground_overlap: float | None = None
    dice_coefficient: float | None = None
    centre_of_mass_distance_mm: float | None = None


class TransformSummary(BaseModel):
    """Persisted transform information."""

    model_config = ConfigDict(extra="forbid")

    transform_type: str
    parameters: list[float]
    fixed_parameters: list[float]
    translation_mm_xyz: tuple[float, float, float]
    translation_voxels_zyx: tuple[float, float, float]
    rotation_degrees: tuple[float, float, float]
    affine_matrix: list[float] | None = None
    affine_scale: tuple[float, float, float] | None = None
    affine_shear: float | None = None


class OptimiserSummary(BaseModel):
    """Optimizer and stage status."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    attempted: bool
    succeeded: bool
    stop_condition: str
    iterations: int
    metric_value: float
    rejected: bool = False
    warnings: list[str] = Field(default_factory=list)


class RegistrationOutputPaths(BaseModel):
    """Registration output paths."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str
    registered_moving_volume: str
    transform: str
    metrics: str
    metadata: str
    report: str
    overlay_mid_axial: str
    difference_mid_axial: str
    fixed_mid_axial: str
    moving_mid_axial: str
    registered_mid_axial: str


class RegistrationResult(BaseModel):
    """Full deterministic registration result."""

    model_config = ConfigDict(extra="forbid")

    registration_run_id: str
    fixed: VolumeRoleMetadata
    moving: VolumeRoleMetadata
    registration_direction: str
    mode: RegistrationMode
    initialisation_method: InitialisationMethod
    optimiser_configuration: dict[str, object]
    optimiser_summaries: list[OptimiserSummary]
    transform: TransformSummary
    metrics_before: RegistrationMetrics
    metrics_after: RegistrationMetrics
    findings: list[RegistrationFinding]
    status: RegistrationStatus
    warnings: list[str]
    output_paths: RegistrationOutputPaths
    checksums: dict[str, str]
    policy_version: str
    generated_at: str
    processing_duration_seconds: float
    recommended_next_action: str
