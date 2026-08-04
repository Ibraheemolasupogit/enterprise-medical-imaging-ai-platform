"""Pydantic contracts for the local governed API."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DISCLAIMER = (
    "This platform is a research and engineering demonstrator. Outputs are intended for technical "
    "evaluation and human review only and must not be used for clinical diagnosis or "
    "patient-management decisions."
)


class APIConfig(BaseModel):
    """Typed Milestone 11 API settings."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1)
    service_name: str = Field(min_length=1)
    service_version: str = Field(min_length=1)
    environment: str = Field(min_length=1)
    host: str
    port: int = Field(gt=0, le=65535)
    log_level: str = Field(min_length=1)
    allowed_input_roots: list[Path] = Field(min_length=1)
    allowed_evidence_roots: list[Path] = Field(min_length=1)
    maximum_request_bytes: int = Field(gt=0)
    maximum_array_bytes: int = Field(gt=0)
    maximum_batch_size: int = Field(gt=0)
    request_timeout_seconds: int = Field(gt=0)
    enable_metrics_endpoint: bool = False
    metrics_access_token: str | None = None
    enable_docs: bool = True
    enable_openapi: bool = True
    require_model_checksums: bool = True
    require_quality_pass: bool = True
    allow_degraded_review: bool = True
    allow_external_bind: bool = False
    allow_threshold_override: bool = False
    segmentation_checkpoint: Path | None = None
    classification_checkpoint: Path | None = None
    classification_calibration: Path | None = None
    classification_threshold_policy: Path | None = None
    longitudinal_config: Path = Path("config/longitudinal.yaml")
    output_directory: Path = Path("ml/experiments/api")

    @field_validator("allowed_input_roots", "allowed_evidence_roots")
    @classmethod
    def roots_safe(cls, value: list[Path]) -> list[Path]:
        for root in value:
            text = str(root)
            if text in {"", "/", "*", "."}:
                raise ValueError("Allowed roots must be specific local directories.")
            if "://" in text:
                raise ValueError("Remote URL roots are not allowed.")
        return value

    @model_validator(mode="after")
    def relationships_valid(self) -> APIConfig:
        if self.host == "0.0.0.0" and not self.allow_external_bind:  # nosec B104
            raise ValueError("0.0.0.0 requires allow_external_bind=true.")
        return self


class ArrayPayload(BaseModel):
    """Compact JSON array payload."""

    model_config = ConfigDict(extra="forbid")

    shape: tuple[int, int, int]
    values: list[float | int]

    @field_validator("shape")
    @classmethod
    def shape_valid(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(value) != 3 or any(axis <= 0 for axis in value):
            raise ValueError("shape must contain three positive dimensions")
        return value


class SegmentationPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    input_path: str | None = None
    array: ArrayPayload | None = None
    spacing_mm: tuple[float, float, float]
    threshold: float | None = Field(default=None, ge=0, le=1)
    persist_output: bool = False
    source_provenance: dict[str, str] = Field(default_factory=dict)

    @field_validator("spacing_mm")
    @classmethod
    def spacing_valid(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(axis <= 0 for axis in value):
            raise ValueError("spacing_mm must contain three positive values")
        return value

    @model_validator(mode="after")
    def exactly_one_input(self) -> SegmentationPredictRequest:
        if (self.input_path is None) == (self.array is None):
            raise ValueError("Provide exactly one of input_path or array.")
        return self


class ClassificationPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    input_path: str | None = None
    array: ArrayPayload | None = None
    source_provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def exactly_one_input(self) -> ClassificationPredictRequest:
        if (self.input_path is None) == (self.array is None):
            raise ValueError("Provide exactly one of input_path or array.")
        return self


class LongitudinalAnalyseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    previous_mask_path: str | None = None
    current_mask_path: str | None = None
    previous_array: ArrayPayload | None = None
    current_array: ArrayPayload | None = None
    previous_spacing_mm: tuple[float, float, float]
    current_spacing_mm: tuple[float, float, float]
    case_id: str
    research_subject_id: str
    side: Literal["left", "right"]
    previous_timepoint: str = "previous"
    current_timepoint: str = "current"
    registration_status: str = "PASS"
    segmentation_status: str = "PASS"
    classification_status: str = "PASS"
    classification_abstention_status: str = "NOT_ABSTAINED"
    registration_run_id: str | None = "api-registration"
    persist_output: bool = True

    @field_validator("previous_spacing_mm", "current_spacing_mm")
    @classmethod
    def spacing_valid(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(axis <= 0 for axis in value):
            raise ValueError("spacing values must contain three positive values")
        return value

    @model_validator(mode="after")
    def input_pairs_valid(self) -> LongitudinalAnalyseRequest:
        path_pair = self.previous_mask_path is not None or self.current_mask_path is not None
        array_pair = self.previous_array is not None or self.current_array is not None
        if path_pair == array_pair:
            raise ValueError("Provide either both mask paths or both arrays.")
        if path_pair and (self.previous_mask_path is None or self.current_mask_path is None):
            raise ValueError("Both previous and current mask paths are required.")
        if array_pair and (self.previous_array is None or self.current_array is None):
            raise ValueError("Both previous and current arrays are required.")
        return self


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_code: str
    message: str
    request_id: str
    details: dict[str, str] = Field(default_factory=dict)
