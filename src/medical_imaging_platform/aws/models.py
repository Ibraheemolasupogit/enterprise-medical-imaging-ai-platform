"""Typed models for AWS infrastructure assurance evidence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AwsStatus = Literal["PASS", "FAIL", "INCOMPLETE", "UNAVAILABLE", "ERROR"]

AWS_DISCLAIMER = (
    "AWS infrastructure evidence for a research and engineering demonstrator only. "
    "Synthetic or public de-identified data only. Not for diagnosis, clinical decision-making, "
    "NHS approval, medical-device claims, automated promotion, automated retraining, or proof "
    "of a live clinical service."
)


class AwsCheckResult(BaseModel):
    """One AWS/Terraform validation result."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    status: AwsStatus
    message: str
    mandatory: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class AwsEvidenceManifest(BaseModel):
    """Machine-readable Milestone 16 AWS evidence manifest."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    generated_timestamp: str
    overall_status: AwsStatus
    terraform_root: str
    architecture_manifest: dict[str, Any]
    terraform_results: list[AwsCheckResult]
    policy_results: list[AwsCheckResult]
    scan_results: list[AwsCheckResult]
    resource_inventory: dict[str, Any]
    iam_permission_summary: dict[str, Any]
    networking_summary: dict[str, Any]
    encryption_summary: dict[str, Any]
    cost_driver_report: list[dict[str, str]]
    checksums: dict[str, str]
    disclaimer: str = AWS_DISCLAIMER
