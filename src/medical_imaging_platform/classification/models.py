"""Typed models for synthetic lesion-presence classification."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ClassificationStatus = Literal["PASS", "PASS_WITH_WARNINGS", "FAIL", "REJECTED"]
SplitName = Literal["train", "validation", "test"]
EngineeringLabel = Literal[
    "no_visible_synthetic_lesion", "synthetic_lesion_present", "indeterminate"
]


class ClassificationConfig(BaseModel):
    """Milestone 9 classification policy."""

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
    channels: tuple[int, ...]
    dropout: float = Field(ge=0.0, lt=1.0)
    positive_class_weight: float | None = Field(default=None, gt=0)
    early_stopping_patience: int = Field(ge=0)
    calibration_method: Literal["none", "platt", "isotonic"] = "platt"
    threshold_method: Literal[
        "fixed", "minimum_sensitivity", "youden", "minimum_npv", "maximum_false_positives"
    ] = "fixed"
    fixed_threshold: float = Field(ge=0.0, le=1.0)
    minimum_sensitivity: float = Field(ge=0.0, le=1.0)
    maximum_false_positives: int = Field(ge=0)
    abstention_lower: float = Field(ge=0.0, le=1.0)
    abstention_upper: float = Field(ge=0.0, le=1.0)
    minimum_validation_auroc: float = Field(ge=0.0, le=1.0)
    minimum_validation_auprc: float = Field(ge=0.0, le=1.0)
    minimum_validation_recall: float = Field(ge=0.0, le=1.0)
    maximum_false_negatives: int = Field(ge=0)
    maximum_brier_score: float = Field(ge=0.0, le=1.0)
    dataset_output_directory: Path = Path("ml/datasets/classification")
    output_directory: Path = Path("ml/experiments/classification")
    overwrite: bool = False

    @field_validator("input_shape")
    @classmethod
    def input_shape_valid(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(value) != 3 or any(axis < 8 for axis in value):
            raise ValueError("input_shape must contain three dimensions of at least 8")
        return value

    @field_validator("channels")
    @classmethod
    def channels_valid(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) < 1 or any(channel <= 0 for channel in value):
            raise ValueError("channels must contain positive values")
        return value

    @model_validator(mode="after")
    def relationships_valid(self) -> ClassificationConfig:
        if self.abstention_lower > self.abstention_upper:
            raise ValueError("abstention_lower must be <= abstention_upper")
        return self


class ClassificationSample(BaseModel):
    """One prepared classification ROI sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    image: str
    label: Literal[0, 1]
    label_name: Literal["no_visible_synthetic_lesion", "synthetic_lesion_present"]
    case_id: str
    research_subject_id: str
    split: SplitName
    scenario: str
    side: Literal["left", "right"]
    timepoint: Literal["previous", "current"]
    spacing_mm: tuple[float, float, float]
    source_checksums: dict[str, str]
    localisation_run_id: str | None = None
    segmentation_run_id: str | None = None


class ClassificationDatasetManifest(BaseModel):
    """Prepared classification dataset manifest."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    source_dataset_id: str
    policy_version: str
    generated_at: str
    input_shape: tuple[int, int, int]
    samples: list[ClassificationSample]
    split_counts: dict[str, int]
    class_counts: dict[str, int]
    checksums: dict[str, str]
    label_semantics: dict[str, str]


class ClassificationFinding(BaseModel):
    """One classification quality finding."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    severity: Literal["INFO", "WARNING", "ERROR", "CRITICAL"]
    status: Literal["PASS", "FAIL", "NOT_EVALUATED"]
    message: str
    observed_value: object | None = None
    expected_value: object | None = None
    remediation: str
