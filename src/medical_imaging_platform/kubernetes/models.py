"""Typed models for Kubernetes deployment assurance evidence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

KubernetesStatus = Literal["PASS", "FAIL", "INCOMPLETE", "UNAVAILABLE", "ERROR"]

KUBERNETES_DISCLAIMER = (
    "Local Kubernetes deployment evidence for a research and engineering demonstrator only. "
    "Synthetic or public de-identified data only. Not for diagnosis, clinical decision-making, "
    "NHS approval, medical-device claims, automated promotion, or automated retraining."
)


class KubernetesCheckResult(BaseModel):
    """One Kubernetes validation result."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    status: KubernetesStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class KubernetesSmokeResult(BaseModel):
    """Local Kubernetes runtime smoke evidence."""

    model_config = ConfigDict(extra="forbid")

    status: KubernetesStatus
    executed: bool
    runtime: str | None = None
    steps: list[KubernetesCheckResult] = Field(default_factory=list)
    cleanup_status: KubernetesStatus = "UNAVAILABLE"
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class KubernetesEvidenceManifest(BaseModel):
    """Machine-readable Milestone 15 Kubernetes evidence manifest."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    generated_timestamp: str
    overall_status: KubernetesStatus
    helm_chart: str
    rendered_manifest: str
    helm_lint: KubernetesCheckResult
    schema_validation: KubernetesCheckResult
    policy_validation: list[KubernetesCheckResult]
    smoke_result: KubernetesSmokeResult
    workload_inventory: list[dict[str, Any]]
    security_controls: dict[str, Any]
    checksums: dict[str, str]
    disclaimer: str = KUBERNETES_DISCLAIMER
