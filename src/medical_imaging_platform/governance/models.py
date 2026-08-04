"""Typed Milestone 14 governance evidence models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LifecycleState = Literal["candidate", "approved", "rejected", "retired"]
MonitoringStatus = Literal["PASS", "WARN", "ALERT"]
ModelType = Literal["segmentation", "classification"]
ReviewerAction = Literal["none", "accepted", "rejected", "corrected", "indeterminate"]
ActorType = Literal["system", "human_reviewer", "governance_reviewer", "service_account"]

RESEARCH_DISCLAIMER = (
    "Synthetic local governance evidence for a research and engineering demonstrator only. "
    "Not for diagnosis, patient management, clinical performance claims, NHS approval, or "
    "medical-device release."
)


class ApprovalMetadata(BaseModel):
    """Explicit human approval metadata."""

    model_config = ConfigDict(extra="forbid")

    approved_by: str = Field(min_length=1)
    approval_ticket: str = Field(min_length=1)
    approval_timestamp: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    actor_type: Literal["governance_reviewer", "human_reviewer"] = "governance_reviewer"


class ModelVersionRecord(BaseModel):
    """One local model registry version record."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    model_type: ModelType
    lifecycle_state: LifecycleState = "candidate"
    checkpoint_reference: str = Field(min_length=1)
    checkpoint_checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    framework_versions: dict[str, str] = Field(min_length=1)
    config_reference: str = Field(min_length=1)
    config_checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    training_data_reference: str = Field(min_length=1)
    evaluation_metrics: dict[str, float] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    calibration_metadata: dict[str, Any] = Field(default_factory=dict)
    approval_metadata: ApprovalMetadata | None = None
    created_timestamp: str = "2026-01-01T00:00:00Z"
    disclaimer: str = RESEARCH_DISCLAIMER

    @model_validator(mode="after")
    def approval_required_for_approved(self) -> ModelVersionRecord:
        if self.lifecycle_state == "approved" and self.approval_metadata is None:
            raise ValueError("Approved model versions require explicit approval metadata.")
        return self


class RegistryManifest(BaseModel):
    """Append-only style local registry manifest."""

    model_config = ConfigDict(extra="forbid")

    registry_version: str = "m14-model-registry-v1"
    records: list[ModelVersionRecord] = Field(default_factory=list)
    disclaimer: str = RESEARCH_DISCLAIMER


class MonitoringThresholds(BaseModel):
    """Simple deterministic monitoring thresholds."""

    model_config = ConfigDict(extra="forbid")

    warn_multiplier: float = Field(gt=0)
    alert_multiplier: float = Field(gt=0)
    alert_severity: str = Field(min_length=1)

    @model_validator(mode="after")
    def alert_exceeds_warn(self) -> MonitoringThresholds:
        if self.alert_multiplier <= self.warn_multiplier:
            raise ValueError("Alert multiplier must exceed warn multiplier.")
        return self


class MetricBaseline(BaseModel):
    """Baseline summary for one synthetic monitoring metric."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    category: str
    mean: float
    std: float
    warn_delta: float
    alert_delta: float


class MonitoringBaseline(BaseModel):
    """Stored deterministic synthetic monitoring baseline."""

    model_config = ConfigDict(extra="forbid")

    baseline_id: str
    generated_timestamp: str
    window_label: str
    thresholds: MonitoringThresholds
    metrics: list[MetricBaseline]
    disclaimer: str = RESEARCH_DISCLAIMER


class CurrentMetric(BaseModel):
    """Current synthetic monitoring metric value."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    category: str
    value: float
    labels_available: bool = False


class DriftFinding(BaseModel):
    """One deterministic drift comparison finding."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str
    category: str
    baseline_mean: float
    current_value: float
    absolute_delta: float
    status: MonitoringStatus
    message: str


class MonitoringRun(BaseModel):
    """Synthetic monitoring run evidence."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    mode: Literal["normal", "simulated_drift"]
    generated_timestamp: str
    overall_status: MonitoringStatus
    metrics: list[CurrentMetric]
    findings: list[DriftFinding]
    disclaimer: str = RESEARCH_DISCLAIMER


class AlertSummary(BaseModel):
    """Aggregated monitoring alert summary."""

    model_config = ConfigDict(extra="forbid")

    summary_id: str
    generated_timestamp: str
    overall_status: MonitoringStatus
    alert_count: int
    warn_count: int
    recommended_action: str
    disclaimer: str = RESEARCH_DISCLAIMER


SENSITIVE_AUDIT_FIELDS = {
    "patient_id",
    "patient_name",
    "nhs_number",
    "mrn",
    "accession_number",
    "date_of_birth",
    "raw_payload",
    "image_array",
}


class AuditEvent(BaseModel):
    """One append-only synthetic audit event."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    config_version: str = Field(min_length=1)
    input_provenance_reference: str = Field(min_length=1)
    output_checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    reviewer_action: ReviewerAction = "none"
    override_event: str | None = None
    export_event: str | None = None
    actor_type: ActorType
    metadata: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = RESEARCH_DISCLAIMER

    @field_validator("metadata")
    @classmethod
    def reject_sensitive_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        forbidden = sorted(SENSITIVE_AUDIT_FIELDS.intersection(value))
        if forbidden:
            raise ValueError(f"Audit metadata contains PHI-sensitive fields: {forbidden}")
        return value
