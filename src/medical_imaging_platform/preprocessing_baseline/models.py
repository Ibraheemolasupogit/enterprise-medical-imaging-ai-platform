"""Typed models for M19 preprocessing baseline evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BaselineTransformName = Literal[
    "noop",
    "mean_denoise",
    "median_denoise",
    "percentile_stretch",
    "histogram_equalization",
    "clahe",
]


class DenoiseConfig(BaseModel):
    """Deterministic intensity-data denoising settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["mean", "median"] = "mean"
    kernel_size: int = Field(default=3, ge=3, le=7)

    @field_validator("kernel_size")
    @classmethod
    def kernel_size_must_be_odd(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("denoise kernel_size must be odd")
        return value


class EnhancementConfig(BaseModel):
    """Deterministic intensity enhancement settings for scalar volumes."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["percentile_stretch", "histogram_equalization", "clahe"] = "percentile_stretch"
    lower_percentile: float = Field(default=1.0, ge=0.0, lt=100.0)
    upper_percentile: float = Field(default=99.0, gt=0.0, le=100.0)
    histogram_bins: int = Field(default=128, ge=16, le=4096)
    clahe_tiles: tuple[int, int] = (4, 4)
    clahe_clip_limit: float = Field(default=0.02, gt=0.0, le=1.0)

    @field_validator("clahe_tiles")
    @classmethod
    def clahe_tiles_must_be_positive(cls, value: tuple[int, int]) -> tuple[int, int]:
        if any(item <= 0 for item in value):
            raise ValueError("clahe_tiles must contain positive values")
        return value

    @model_validator(mode="after")
    def percentiles_must_be_ordered(self) -> EnhancementConfig:
        if self.lower_percentile >= self.upper_percentile:
            raise ValueError("lower_percentile must be less than upper_percentile")
        return self


class MetricSettings(BaseModel):
    """Image-content metric settings."""

    model_config = ConfigDict(extra="forbid")

    percentiles: tuple[float, ...] = (1.0, 5.0, 50.0, 95.0, 99.0)
    histogram_bins: int = Field(default=128, ge=16, le=4096)
    edge_percentile: float = Field(default=90.0, gt=0.0, lt=100.0)
    local_contrast_kernel_size: int = Field(default=3, ge=3, le=7)

    @field_validator("percentiles")
    @classmethod
    def percentiles_must_be_valid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("at least one percentile is required")
        if any(item < 0.0 or item > 100.0 for item in value):
            raise ValueError("percentiles must be between 0 and 100")
        return value

    @field_validator("local_contrast_kernel_size")
    @classmethod
    def local_contrast_kernel_size_must_be_odd(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("local_contrast_kernel_size must be odd")
        return value


class ThresholdSettings(BaseModel):
    """Optional evidence-only metric thresholds."""

    model_config = ConfigDict(extra="forbid")

    minimum_sharpness: float | None = Field(default=None, ge=0.0)
    minimum_edge_density: float | None = Field(default=None, ge=0.0, le=1.0)
    maximum_noise_estimate: float | None = Field(default=None, ge=0.0)


class PreviewSettings(BaseModel):
    """Representative preview settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    axial_slice_index: int | None = Field(default=None, ge=0)


class PreprocessingBaselineConfig(BaseModel):
    """Configuration for M19 classical preprocessing baseline evidence."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1)
    baseline_name: str = Field(default="classical-image-content-baseline", min_length=1)
    denoise: DenoiseConfig = Field(default_factory=DenoiseConfig)
    enhancement: EnhancementConfig = Field(default_factory=EnhancementConfig)
    metrics: MetricSettings = Field(default_factory=MetricSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    preview: PreviewSettings = Field(default_factory=PreviewSettings)
    output_directory: Path = Path("reports/preprocessing_baseline")
    overwrite: bool = False


class ImageContentMetrics(BaseModel):
    """Deterministic image-content metric payload."""

    model_config = ConfigDict(extra="forbid")

    shape: tuple[int, ...]
    dtype: str
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    percentiles: dict[str, float]
    sharpness_laplacian_variance: float
    noise_high_frequency_residual_std: float
    edge_density: float
    entropy: float
    local_contrast_mean: float


class TransformRecord(BaseModel):
    """One applied classical transform."""

    model_config = ConfigDict(extra="forbid")

    name: BaselineTransformName
    parameters: dict[str, object]


class BaselineSourceEvidence(BaseModel):
    """Before/after evidence for one preprocessed source volume."""

    model_config = ConfigDict(extra="forbid")

    source_output_dir: str
    source_preprocessing_run_id: str
    source_metadata_checksum: str | None
    source_volume_checksum: str | None
    source_report_checksum: str | None
    volume_shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    before_metrics: ImageContentMetrics
    after_metrics: ImageContentMetrics
    metric_deltas: dict[str, float]
    transform_chain: list[TransformRecord]
    transformed_volume: str
    preview_before: str | None
    preview_after: str | None
    warnings: list[str]


class BaselineOutputPaths(BaseModel):
    """Paths written by one M19 baseline run."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str
    metrics: str
    summary: str
    report: str
    checksum_manifest: str


class PreprocessingBaselineResult(BaseModel):
    """Top-level M19 baseline result."""

    model_config = ConfigDict(extra="forbid")

    baseline_run_id: str
    policy_version: str
    baseline_name: str
    generated_at: str
    source_count: int
    sources: list[BaselineSourceEvidence]
    monai_comparison: dict[str, object]
    output_paths: BaselineOutputPaths
    checksums: dict[str, str]
    warnings: list[str]
    limitations: list[str]
