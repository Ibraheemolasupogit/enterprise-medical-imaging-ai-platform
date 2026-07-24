"""Typed models for Milestone 8 synthetic lesion segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SegmentationStatus = Literal["PASS", "PASS_WITH_WARNINGS", "FAIL", "REJECTED"]
SplitName = Literal["train", "validation", "test"]


class AugmentationConfig(BaseModel):
    """Lightweight CPU-safe transform settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    flip_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    noise_std: float = Field(default=0.01, ge=0.0)


class SegmentationConfig(BaseModel):
    """Milestone 8 segmentation experiment policy."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1)
    random_seed: int = Field(ge=0)
    device: Literal["cpu", "cuda"] = "cpu"
    input_shape: tuple[int, int, int]
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    optimizer: Literal["adam", "adamw"] = "adam"
    loss: Literal["dice", "bce", "dice_bce"] = "dice_bce"
    dice_weight: float = Field(ge=0)
    bce_weight: float = Field(ge=0)
    channels: tuple[int, ...]
    strides: tuple[int, ...]
    num_res_units: int = Field(ge=0)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    early_stopping_patience: int = Field(ge=0)
    threshold: float = Field(gt=0.0, lt=1.0)
    minimum_component_voxels: int = Field(ge=0)
    keep_largest_component: bool = False
    fill_holes: bool = True
    minimum_validation_dice: float = Field(ge=0.0, le=1.0)
    minimum_test_recall: float = Field(ge=0.0, le=1.0)
    maximum_false_positive_voxels: int = Field(ge=0)
    maximum_relative_volume_error: float = Field(ge=0.0)
    dataset_output_directory: Path = Path("ml/datasets/segmentation")
    output_directory: Path = Path("ml/experiments/segmentation")
    overwrite: bool = False

    @field_validator("input_shape")
    @classmethod
    def input_shape_valid(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(value) != 3 or any(axis < 8 for axis in value):
            raise ValueError("input_shape must have three dimensions of at least 8")
        return value

    @field_validator("channels")
    @classmethod
    def channels_valid(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) < 2 or any(channel <= 0 for channel in value):
            raise ValueError("channels must contain at least two positive values")
        return value

    @field_validator("strides")
    @classmethod
    def strides_valid(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(stride <= 0 for stride in value):
            raise ValueError("strides must contain positive values")
        return value

    @model_validator(mode="after")
    def combined_settings_valid(self) -> SegmentationConfig:
        if len(self.strides) != len(self.channels) - 1:
            raise ValueError("strides length must equal channels length minus one")
        if self.loss == "dice" and self.dice_weight <= 0:
            raise ValueError("dice loss requires positive dice_weight")
        if self.loss == "bce" and self.bce_weight <= 0:
            raise ValueError("bce loss requires positive bce_weight")
        if self.loss == "dice_bce" and self.dice_weight + self.bce_weight <= 0:
            raise ValueError("combined loss requires at least one positive weight")
        return self


class SegmentationSample(BaseModel):
    """One segmentation-ready sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    image: str
    lesion_mask: str
    case_id: str
    research_subject_id: str
    split: SplitName
    scenario: str
    timepoint: Literal["previous", "current"]
    spacing_mm: tuple[float, float, float]
    source_checksums: dict[str, str]
    localisation_run_id: str | None = None
    lesion_volume_voxels: int = Field(ge=0)


class SegmentationDatasetManifest(BaseModel):
    """Prepared segmentation dataset manifest."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    source_dataset_id: str
    policy_version: str
    generated_at: str
    input_shape: tuple[int, int, int]
    samples: list[SegmentationSample]
    split_counts: dict[str, int]
    checksums: dict[str, str]


class SegmentationMetricSet(BaseModel):
    """Voxel and surface segmentation metrics."""

    model_config = ConfigDict(extra="forbid")

    dice: float | None
    iou: float | None
    precision: float | None
    recall: float | None
    sensitivity: float | None
    specificity: float | None
    false_positive_voxels: int
    false_negative_voxels: int
    predicted_lesion_volume_voxels: int
    ground_truth_lesion_volume_voxels: int
    absolute_volume_error_voxels: int
    relative_volume_error: float | None
    hausdorff95_mm: float | None
    average_surface_distance_mm: float | None
    both_masks_empty: bool


class SegmentationFinding(BaseModel):
    """One model-quality finding."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    severity: Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
    status: Literal["PASS", "FAIL", "NOT_EVALUATED"]
    message: str
    observed_value: object | None = None
    expected_value: object | None = None
    remediation: str
