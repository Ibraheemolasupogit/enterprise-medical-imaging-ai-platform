"""Typed portfolio evidence models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PortfolioStatus = Literal["PASS", "WARN", "FAIL", "INCOMPLETE", "UNAVAILABLE", "ERROR"]
ClaimClassification = Literal[
    "implemented",
    "locally executed",
    "statically validated",
    "simulated",
    "target-state only",
]

PORTFOLIO_DISCLAIMER = (
    "Portfolio evidence for a research and engineering demonstrator only. Synthetic or public "
    "de-identified data only. Not for diagnosis, clinical decision-making, NHS approval, "
    "medical-device claims, live clinical deployment, automated model promotion, automated "
    "retraining, or automated rollback."
)


class PortfolioCheck(BaseModel):
    """One portfolio validation result."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    status: PortfolioStatus
    message: str
    mandatory: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class MilestoneEntry(BaseModel):
    """Portfolio milestone matrix row."""

    model_config = ConfigDict(extra="forbid")

    milestone: int
    title: str
    principal_capability: str
    implementation_status: PortfolioStatus
    evidence_path: str
    validation_command: str
    test_coverage: str
    known_limitations: list[str]
    deployment_status: str
    claim_classification: ClaimClassification


class PortfolioEvidenceManifest(BaseModel):
    """Final machine-readable portfolio evidence manifest."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    generated_timestamp: str
    overall_status: PortfolioStatus
    repository_reference: dict[str, str]
    status_semantics: dict[str, str]
    milestone_completion_matrix: list[MilestoneEntry]
    capability_inventory: list[dict[str, Any]]
    architecture_manifest: dict[str, Any]
    technology_inventory: list[dict[str, str]]
    validation_summary: list[PortfolioCheck]
    test_and_coverage_summary: dict[str, Any]
    security_control_summary: list[dict[str, str]]
    governance_control_summary: list[dict[str, str]]
    model_performance_summary: list[dict[str, Any]]
    deployment_assurance_summary: list[dict[str, Any]]
    observability_operations_summary: list[dict[str, Any]]
    limitations_summary: list[str]
    evidence_provenance: dict[str, Any]
    checksums: dict[str, str]
    disclaimer: str = PORTFOLIO_DISCLAIMER
