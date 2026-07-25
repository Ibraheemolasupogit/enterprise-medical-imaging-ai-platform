"""Typed contracts for the reviewer UI."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from medical_imaging_platform.api.models import DISCLAIMER

ReviewerDecisionValue = Literal[
    "accepted_for_engineering_review",
    "needs_secondary_review",
    "rejected_due_to_quality",
    "insufficient_information",
]
EvidenceType = Literal["segmentation", "classification", "longitudinal"]


class ReviewerUIConfig(BaseModel):
    """Typed Milestone 12 reviewer UI settings."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1)
    page_title: str = Field(min_length=1)
    page_icon: str = Field(min_length=1)
    layout: Literal["centered", "wide"] = "wide"
    api_base_url: str = "http://127.0.0.1:8000"
    request_timeout_seconds: int = Field(gt=0)
    allowed_upload_extensions: list[str] = Field(min_length=1)
    maximum_upload_bytes: int = Field(gt=0)
    maximum_review_items: int = Field(gt=0)
    enable_segmentation_page: bool = True
    enable_classification_page: bool = True
    enable_longitudinal_page: bool = True
    enable_evidence_page: bool = True
    enable_governance_page: bool = True
    allow_local_array_upload: bool = True
    allow_evidence_export: bool = True
    allow_remote_api: bool = False
    allow_remote_bind: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8501, gt=0, le=65535)
    review_output_directory: Path = Path("reports/generated/reviewer-sessions")
    show_debug_details: bool = False

    @field_validator("allowed_upload_extensions")
    @classmethod
    def extensions_valid(cls, value: list[str]) -> list[str]:
        allowed = {".npy", ".json"}
        normalised = [extension.lower() for extension in value]
        if any(extension not in allowed for extension in normalised):
            raise ValueError("Reviewer UI uploads are limited to .npy and .json files.")
        return normalised

    @field_validator("review_output_directory")
    @classmethod
    def output_directory_safe(cls, value: Path) -> Path:
        text = str(value)
        if text in {"", "/", "*", "."} or "://" in text:
            raise ValueError("Review output directory must be a specific local path.")
        return value

    @model_validator(mode="after")
    def network_scope_valid(self) -> ReviewerUIConfig:
        if not self.allow_remote_api and not _is_loopback_url(self.api_base_url):
            raise ValueError("Remote API URLs require allow_remote_api=true.")
        if self.host == "0.0.0.0" and not self.allow_remote_bind:  # nosec B104
            raise ValueError("0.0.0.0 requires allow_remote_bind=true.")
        return self


class ReviewerAPIError(Exception):
    """Sanitised API-client error for UI display."""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        request_id: str | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.request_id = request_id
        self.details = details or {}

    def to_display(self) -> dict[str, object]:
        return {
            "status_code": self.status_code,
            "error_code": self.error_code,
            "message": self.message,
            "request_id": self.request_id,
            "details": self.details,
        }


class ReviewerDecision(BaseModel):
    """Human reviewer decision kept separate from model output."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    request_id: str
    evidence_type: EvidenceType
    evidence_id: str
    model_engineering_label: str
    quality_status: str
    reviewer_decision: ReviewerDecisionValue
    review_notes: str = Field(default="", max_length=500)
    review_timestamp: str
    application_version: str

    @field_validator("review_notes")
    @classmethod
    def notes_safe(cls, value: str) -> str:
        lowered = value.lower()
        forbidden = ("patient", "dob", "nhs", "mrn", "name:")
        if any(token in lowered for token in forbidden):
            raise ValueError("Reviewer notes must not include personal or patient identifiers.")
        return value

    @field_validator("evidence_id", "review_id")
    @classmethod
    def identifier_safe(cls, value: str) -> str:
        if "/" in value or "\\" in value or ".." in value or not value.strip():
            raise ValueError("Review identifiers must be local safe identifiers.")
        return value


class ReviewExportResult(BaseModel):
    """Review export result for UI status display."""

    model_config = ConfigDict(extra="forbid")

    output_directory: str
    checksums: dict[str, str]
    files: dict[str, str]


def create_timestamp() -> str:
    """Return an ISO timestamp for deterministic schema shape."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def disclaimer() -> str:
    return DISCLAIMER


def _is_loopback_url(url: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def without_sensitive_values(payload: dict[str, Any]) -> dict[str, Any]:
    """Return shallow payload with fields safe for UI display."""
    blocked = {"values", "array", "previous_array", "current_array", "model_weights"}
    return {key: value for key, value in payload.items() if key not in blocked}
