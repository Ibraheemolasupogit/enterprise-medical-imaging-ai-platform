"""Typed operations evidence models for Milestone 17."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

OperationsStatus = Literal["PASS", "WARN", "ALERT", "FAIL", "INCOMPLETE", "UNAVAILABLE", "ERROR"]
IncidentState = Literal[
    "detected",
    "acknowledged",
    "investigating",
    "contained",
    "recovering",
    "resolved",
    "post-incident-review-required",
    "closed",
]

OPERATIONS_DISCLAIMER = (
    "Operations evidence for a research and engineering demonstrator only. Synthetic or public "
    "de-identified data only. Not for diagnosis, clinical decision-making, NHS approval, "
    "medical-device claims, automated deployment, automated retraining, automated model promotion, "
    "or automated production rollback."
)


class OperationsCheck(BaseModel):
    """One operations validation result."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    status: OperationsStatus
    message: str
    mandatory: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class IncidentRecord(BaseModel):
    """Deterministic incident simulation record."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str
    severity: OperationsStatus
    trigger: str
    affected_component: str
    timeline: list[dict[str, str]]
    detection_evidence: str
    containment_action: str
    recovery_action: str
    verification_result: str
    owner_role: str
    status: IncidentState
    lessons_next_action: str


class OperationsEvidenceManifest(BaseModel):
    """Machine-readable operations evidence manifest."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    generated_timestamp: str
    overall_status: OperationsStatus
    observability_manifest: dict[str, Any]
    metrics_inventory: list[dict[str, Any]]
    structured_log_validation: list[OperationsCheck]
    slo_evaluation: list[OperationsCheck]
    error_budget_report: dict[str, Any]
    normal_operations_run: dict[str, Any]
    simulated_incidents: list[IncidentRecord]
    incident_lifecycle_records: list[dict[str, Any]]
    rollback_evidence: dict[str, Any]
    recovery_verification: list[OperationsCheck]
    runbook_inventory: list[dict[str, Any]]
    resilience_control_summary: dict[str, Any]
    checksums: dict[str, str]
    disclaimer: str = OPERATIONS_DISCLAIMER
