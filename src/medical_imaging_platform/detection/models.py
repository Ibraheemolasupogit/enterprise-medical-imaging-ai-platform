"""Typed contracts for synthetic 2D lesion detection."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DetectionStatus = Literal["PASS", "PASS_WITH_WARNINGS", "FAIL", "REJECTED"]
DetectionModelFamily = Literal["faster_rcnn", "tiny_detr"]
SplitName = Literal["train", "validation", "test"]


class DetectionConfig(BaseModel):
    """Milestone 21 detection policy."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1)
    random_seed: int = Field(ge=0)
    device: Literal["cpu", "cuda"] = "cpu"
    model_family: DetectionModelFamily = "faster_rcnn"
    image_shape: tuple[int, int]
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    optimizer: Literal["adam", "adamw"] = "adam"
    max_positive_slices_per_volume: int = Field(gt=0)
    max_negative_slices_per_volume: int = Field(ge=0)
    minimum_lesion_pixels: int = Field(gt=0)
    score_threshold: float = Field(ge=0.0, le=1.0)
    nms_iou_threshold: float = Field(ge=0.0, le=1.0)
    evaluation_iou_thresholds: tuple[float, ...] = (0.5, 0.75)
    faster_rcnn_anchor_sizes: tuple[int, ...] = (8, 16)
    faster_rcnn_aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    hidden_channels: int = Field(default=16, gt=0)
    embedding_dim: int = Field(default=32, gt=0)
    transformer_depth: int = Field(default=1, gt=0)
    num_heads: int = Field(default=4, gt=0)
    num_queries: int = Field(default=4, gt=0)
    mlp_ratio: float = Field(default=2.0, gt=0)
    no_object_weight: float = Field(default=0.1, gt=0)
    box_loss_weight: float = Field(default=5.0, gt=0)
    iou_loss_weight: float = Field(default=2.0, ge=0)
    minimum_validation_map: float = Field(default=0.0, ge=0.0, le=1.0)
    maximum_false_positives_per_image: float = Field(default=100.0, ge=0.0)
    dataset_output_directory: Path = Path("ml/datasets/detection")
    output_directory: Path = Path("ml/experiments/detection")
    overwrite: bool = False

    @field_validator("image_shape")
    @classmethod
    def image_shape_valid(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or any(axis < 8 for axis in value):
            raise ValueError("image_shape must contain height and width of at least 8")
        return value

    @field_validator("evaluation_iou_thresholds")
    @classmethod
    def thresholds_valid(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(threshold <= 0 or threshold > 1 for threshold in value):
            raise ValueError("evaluation_iou_thresholds must be in (0, 1]")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def relationships_valid(self) -> DetectionConfig:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        return self


class DetectionSample(BaseModel):
    """One prepared 2D detection sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    image: str
    boxes: list[list[float]]
    labels: list[int]
    case_id: str
    research_subject_id: str
    split: SplitName
    scenario: str
    timepoint: Literal["previous", "current"]
    source_z: int = Field(ge=0)
    source_volume_shape: tuple[int, int, int]
    image_shape: tuple[int, int]
    spacing_mm: tuple[float, float, float]
    source_checksums: dict[str, str]
    positive: bool

    @model_validator(mode="after")
    def boxes_and_labels_match(self) -> DetectionSample:
        if len(self.boxes) != len(self.labels):
            raise ValueError("boxes and labels must have equal length")
        height, width = self.image_shape
        for box in self.boxes:
            if len(box) != 4:
                raise ValueError("boxes must be xyxy lists of length 4")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("boxes must be ordered and within image bounds")
        if any(label != 1 for label in self.labels):
            raise ValueError("synthetic lesion detection labels must be 1")
        if self.positive != bool(self.boxes):
            raise ValueError("positive flag must match presence of boxes")
        return self


class DetectionDatasetManifest(BaseModel):
    """Prepared detection dataset manifest."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    source_dataset_id: str
    policy_version: str
    generated_at: str
    image_shape: tuple[int, int]
    coordinate_semantics: str
    samples: list[DetectionSample]
    split_counts: dict[str, int]
    positive_image_count: int
    negative_image_count: int
    box_count: int
    checksums: dict[str, str]
    label_semantics: dict[str, str]
