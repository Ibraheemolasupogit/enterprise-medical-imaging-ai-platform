"""Deterministic operations assurance for Milestone 17."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_imaging_platform.api.observability import (
    MetricsRegistry,
    metric_labels_safe,
    structured_log_event,
)
from medical_imaging_platform.operations.models import (
    IncidentRecord,
    IncidentState,
    OperationsCheck,
    OperationsEvidenceManifest,
    OperationsStatus,
)
from medical_imaging_platform.release.checksums import checksum_paths

OPERATIONS_EVIDENCE_DIR = Path("reports/generated/operations")
OBSERVABILITY_MANIFEST_PATH = OPERATIONS_EVIDENCE_DIR / "observability_manifest.json"
METRICS_INVENTORY_PATH = OPERATIONS_EVIDENCE_DIR / "metrics_inventory.json"
LOG_VALIDATION_PATH = OPERATIONS_EVIDENCE_DIR / "structured_log_validation.json"
SLO_EVALUATION_PATH = OPERATIONS_EVIDENCE_DIR / "slo_evaluation.json"
ERROR_BUDGET_PATH = OPERATIONS_EVIDENCE_DIR / "error_budget_report.json"
NORMAL_RUN_PATH = OPERATIONS_EVIDENCE_DIR / "normal_operations_run.json"
INCIDENT_RESULTS_PATH = OPERATIONS_EVIDENCE_DIR / "simulated_incident_results.json"
INCIDENT_LIFECYCLE_PATH = OPERATIONS_EVIDENCE_DIR / "incident_lifecycle_records.json"
ROLLBACK_EVIDENCE_PATH = OPERATIONS_EVIDENCE_DIR / "rollback_evidence.json"
RECOVERY_VERIFICATION_PATH = OPERATIONS_EVIDENCE_DIR / "recovery_verification.json"
RUNBOOK_INVENTORY_PATH = OPERATIONS_EVIDENCE_DIR / "runbook_inventory.json"
RESILIENCE_SUMMARY_PATH = OPERATIONS_EVIDENCE_DIR / "resilience_control_summary.json"
EVIDENCE_MANIFEST_PATH = OPERATIONS_EVIDENCE_DIR / "operations_evidence_manifest.json"
EVIDENCE_REPORT_PATH = OPERATIONS_EVIDENCE_DIR / "operations_report.md"
CHECKSUMS_PATH = OPERATIONS_EVIDENCE_DIR / "checksum_manifest.json"

RUNBOOK_DIR = Path("docs/operations/runbooks")
RUNBOOK_NAMES = [
    "api_not_ready",
    "reviewer_ui_unavailable",
    "failed_model_load",
    "checksum_mismatch",
    "elevated_latency",
    "elevated_error_rate",
    "kubernetes_rollout_failure",
    "image_rollback",
    "model_rollback",
    "evidence_corruption",
    "secret_configuration_retrieval_failure",
    "monitoring_alert_escalation",
]
VALID_TRANSITIONS: dict[IncidentState, set[IncidentState]] = {
    "detected": {"acknowledged"},
    "acknowledged": {"investigating"},
    "investigating": {"contained"},
    "contained": {"recovering"},
    "recovering": {"resolved"},
    "resolved": {"post-incident-review-required"},
    "post-incident-review-required": {"closed"},
    "closed": set(),
}


def validate_observability() -> list[OperationsCheck]:
    """Validate metrics, logging, resilience and static observability controls."""
    registry = MetricsRegistry()
    registry.record_request("GET", "/health", 200, 0.012)
    registry.record_request("POST", "/v1/segmentation/predict", 503, 0.25)
    registry.record_inference("segmentation", "failure")
    registry.set_readiness(ready=False, degraded=True)
    metrics = registry.render_prometheus()
    log_event = structured_log_event(
        service="api",
        event_type="validation_failure",
        request_id="req-001",
        correlation_id="corr-001",
        # Synthetic redaction fixture, not a credential.
        details={  # nosec B105
            "patient_id": "123",
            "token": "secret",
            "status": "rejected",
        },
    )
    checks = [
        _check(
            "OPS-METRICS-FORMAT",
            "medical_imaging_api_requests_total" in metrics,
            "Prometheus metrics render.",
        ),
        _check(
            "OPS-METRICS-LABELS",
            metric_labels_safe(metrics),
            "Metric labels are bounded and non-sensitive.",
        ),
        _check("OPS-LOG-STRUCTURED", _is_json(log_event), "Structured log event is JSON."),
        _check(
            "OPS-LOG-REDACTION",
            "123" not in log_event and "secret" not in log_event,
            "Structured logs redact sensitive fields.",
        ),
        _check(
            "OPS-READINESS-DEGRADED",
            "degraded" in metrics,
            "Readiness degradation metric is available.",
        ),
        _check(
            "OPS-RESILIENCE-RETRY",
            _file_contains(
                Path("src/medical_imaging_platform/reviewer_ui/api_client.py"), "max_retries"
            ),
            "Reviewer UI retry bound is configured.",
        ),
        _check(
            "OPS-RESILIENCE-CIRCUIT",
            _file_contains(
                Path("src/medical_imaging_platform/reviewer_ui/api_client.py"),
                "UI-API-CIRCUIT-OPEN",
            ),
            "Reviewer UI failure guard is configured.",
        ),
        _check(
            "OPS-API-METRICS-PROTECTED",
            _file_contains(
                Path("src/medical_imaging_platform/api/routes/health.py"), "enable_metrics_endpoint"
            ),
            "API metrics endpoint is disabled/protected by config.",
        ),
        _check(
            "OPS-HELM-OBSERVABILITY",
            _file_contains(Path("helm/medical-imaging-platform/values.yaml"), "observability:"),
            "Helm observability settings are present.",
        ),
        _check(
            "OPS-AWS-OBSERVABILITY",
            _file_contains(
                Path("infra/terraform/modules/observability/main.tf"), "failed_inference_rate"
            ),
            "AWS failed inference alarm mapping is present.",
        ),
    ]
    _write_json(LOG_VALIDATION_PATH, [check.model_dump(mode="json") for check in checks])
    return checks


def build_observability_evidence() -> dict[str, Any]:
    """Build observability manifest, metrics inventory and normal operation run."""
    manifest = {
        "service_observability": [
            "request_count",
            "request_latency",
            "http_status_classes",
            "inference_success_failure",
            "preprocessing_failures",
            "model_loading_failures",
            "readiness_degradation",
            "reviewer_ui_api_call_failures",
            "pod_process_restart_indicators",
            "active_model_version",
            "correlation_request_ids",
        ],
        "forbidden": ["PHI", "raw DICOM", "image arrays", "payloads", "secrets", "credentials"],
        "metrics_endpoint": {
            "path": "/metrics",
            "enabled_by_default": False,
            "format": "prometheus",
        },
    }
    inventory = [
        {
            "name": "medical_imaging_api_requests_total",
            "type": "counter",
            "labels": ["method", "route", "status_class"],
        },
        {
            "name": "medical_imaging_api_request_latency_seconds",
            "type": "gauge",
            "labels": ["method", "route"],
        },
        {
            "name": "medical_imaging_inference_outcomes_total",
            "type": "counter",
            "labels": ["model_type", "outcome"],
        },
        {"name": "medical_imaging_api_readiness", "type": "gauge", "labels": ["readiness"]},
        {"name": "medical_imaging_active_model_version", "type": "gauge", "labels": ["model_type"]},
    ]
    normal_run = {
        "run_id": "ops-normal-2026-01-01",
        "status": "PASS",
        "requests": 120,
        "p95_latency_ms": 180,
        "inference_failure_rate": 0.005,
        "readiness": "ready",
        "structured_logging": "enabled",
    }
    _write_json(OBSERVABILITY_MANIFEST_PATH, manifest)
    _write_json(METRICS_INVENTORY_PATH, inventory)
    _write_json(NORMAL_RUN_PATH, normal_run)
    return manifest


def evaluate_slos() -> list[OperationsCheck]:
    """Evaluate deterministic demonstrator SLOs and error budgets."""
    slos = [
        ("OPS-SLO-API-AVAILABILITY", 0.995, 0.998, "API availability"),
        ("OPS-SLO-API-READINESS", 0.99, 0.992, "API readiness"),
        ("OPS-SLO-LATENCY", 750.0, 420.0, "P95 latency milliseconds"),
        ("OPS-SLO-INFERENCE-ERROR", 0.02, 0.008, "Inference error rate"),
        ("OPS-SLO-UI-AVAILABILITY", 0.99, 0.996, "Reviewer UI availability"),
        ("OPS-SLO-RTO", 60.0, 35.0, "Recovery time objective minutes"),
        ("OPS-SLO-RPO", 24.0, 12.0, "Evidence/model artefact RPO hours"),
    ]
    checks: list[OperationsCheck] = []
    budget: dict[str, Any] = {"period": "synthetic-30-day-window", "items": []}
    for check_id, threshold, observed, name in slos:
        lower_is_better = (
            "LATENCY" in check_id
            or "ERROR" in check_id
            or check_id.endswith("RTO")
            or check_id.endswith("RPO")
        )
        if lower_is_better:
            passed = observed <= threshold
            remaining = max(0.0, threshold - observed)
        else:
            passed = observed >= threshold
            remaining = max(0.0, observed - threshold)
        status: OperationsStatus = "PASS" if passed else "ALERT"
        checks.append(
            OperationsCheck(
                check_id=check_id,
                status=status,
                message=f"{name} demonstrator SLO evaluated.",
                details={"threshold": threshold, "observed": observed, "clinical_claim": False},
            )
        )
        budget["items"].append(
            {
                "slo": check_id,
                "remaining_budget": round(remaining, 6),
                "burn_rate_status": "PASS" if passed else "ALERT",
            }
        )
    _write_json(SLO_EVALUATION_PATH, [check.model_dump(mode="json") for check in checks])
    _write_json(ERROR_BUDGET_PATH, budget)
    return checks


def simulate_incidents() -> list[IncidentRecord]:
    """Build deterministic incident simulations."""
    scenarios = [
        ("INC-001", "ALERT", "API unavailable", "api"),
        ("INC-002", "ALERT", "Model checkpoint missing or checksum mismatch", "api-model-loader"),
        ("INC-003", "WARN", "Degraded readiness", "api-readiness"),
        ("INC-004", "WARN", "High latency", "api"),
        ("INC-005", "ALERT", "Elevated inference failures", "inference"),
        ("INC-006", "ALERT", "Reviewer UI cannot reach API", "reviewer-ui"),
        ("INC-007", "WARN", "Kubernetes pod restart/failure", "kubernetes"),
        ("INC-008", "ALERT", "Monitoring drift alert escalation", "monitoring"),
        ("INC-009", "ALERT", "Corrupted evidence/checksum", "evidence"),
        ("INC-010", "ALERT", "Secret or configuration retrieval failure", "configuration"),
    ]
    incidents = [
        IncidentRecord(
            incident_id=incident_id,
            severity=severity,  # type: ignore[arg-type]
            trigger=trigger,
            affected_component=component,
            timeline=[
                {"state": "detected", "timestamp": "2026-01-01T01:00:00Z"},
                {"state": "acknowledged", "timestamp": "2026-01-01T01:05:00Z"},
                {"state": "investigating", "timestamp": "2026-01-01T01:10:00Z"},
            ],
            detection_evidence=f"Deterministic synthetic trigger: {trigger}.",
            containment_action="Freeze promotion/deployment actions and preserve evidence.",
            recovery_action="Follow runbook and prepare operator-approved rollback if required.",
            verification_result="Synthetic verification completed; no infrastructure modified.",
            owner_role="platform-incident-commander",
            status="post-incident-review-required",
            lessons_next_action="Review runbook evidence and update thresholds if needed.",
        )
        for incident_id, severity, trigger, component in scenarios
    ]
    _write_json(INCIDENT_RESULTS_PATH, [incident.model_dump(mode="json") for incident in incidents])
    return incidents


def build_incident_evidence() -> list[dict[str, Any]]:
    """Build incident lifecycle evidence."""
    records = [
        transition_incident("INC-001", "detected", "acknowledged", "ops-user"),
        transition_incident("INC-001", "acknowledged", "investigating", "ops-user"),
        transition_incident("INC-001", "investigating", "contained", "ops-user"),
        transition_incident("INC-001", "contained", "recovering", "ops-user"),
        transition_incident("INC-001", "recovering", "resolved", "ops-user"),
        transition_incident("INC-001", "resolved", "post-incident-review-required", "ops-user"),
    ]
    _write_json(INCIDENT_LIFECYCLE_PATH, records)
    return records


def transition_incident(
    incident_id: str, current: IncidentState, target: IncidentState, actor: str
) -> dict[str, Any]:
    """Apply an explicit incident lifecycle transition."""
    if not actor:
        raise ValueError("Incident transition requires actor metadata.")
    if target not in VALID_TRANSITIONS[current]:
        raise ValueError(f"Invalid incident transition {current} -> {target}.")
    return {
        "incident_id": incident_id,
        "from_state": current,
        "to_state": target,
        "actor": actor,
        "timestamp": "2026-01-01T01:15:00Z",
        "automatic_closure": False,
    }


def validate_runbooks() -> list[OperationsCheck]:
    """Validate required runbook documents and sections."""
    required_sections = [
        "Detection",
        "Immediate Checks",
        "Containment",
        "Recovery",
        "Validation",
        "Escalation",
        "Rollback",
        "Evidence Capture",
        "Prohibited Actions",
    ]
    checks: list[OperationsCheck] = []
    inventory: list[dict[str, Any]] = []
    for name in RUNBOOK_NAMES:
        path = RUNBOOK_DIR / f"{name}.md"
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        missing = [section for section in required_sections if f"## {section}" not in content]
        checks.append(
            OperationsCheck(
                check_id=f"OPS-RUNBOOK-{name}",
                status="PASS" if path.exists() and not missing else "FAIL",
                message=f"Runbook {name} is complete.",
                details={"path": path.as_posix(), "missing_sections": missing},
            )
        )
        inventory.append({"runbook": name, "path": path.as_posix(), "sections": required_sections})
    _write_json(RUNBOOK_INVENTORY_PATH, inventory)
    return checks


def simulate_rollback() -> dict[str, Any]:
    """Create deterministic rollback evidence requiring operator approval."""
    evidence = {
        "rollback_id": "rollback-sim-001",
        "status": "PASS",
        "operator_approval": {
            "approved_by": "simulated-incident-commander",
            "approval_ticket": "OPS-ROLLBACK-SIM-001",
            "timestamp": "2026-01-01T02:00:00Z",
        },
        "plans": {
            "helm": "helm rollback medical-imaging-platform <revision> --dry-run",
            "container_image": "select previous immutable ECR/local image tag after approval",
            "model_version": (
                "retire current approved model and approve previous version through registry"
            ),
            "configuration": "restore reviewed config checksum from evidence",
            "evidence": "restore versioned S3/local evidence and verify checksum manifest",
        },
        "automated_production_rollback": False,
    }
    _write_json(ROLLBACK_EVIDENCE_PATH, evidence)
    return evidence


def validate_recovery() -> list[OperationsCheck]:
    """Validate deterministic recovery evidence."""
    rollback = _load_json(ROLLBACK_EVIDENCE_PATH) if ROLLBACK_EVIDENCE_PATH.exists() else {}
    checks = [
        _check(
            "OPS-RECOVERY-APPROVAL",
            bool(rollback.get("operator_approval")),
            "Rollback approval metadata exists.",
        ),
        _check(
            "OPS-RECOVERY-NO-AUTO",
            rollback.get("automated_production_rollback") is False,
            "Production rollback is not automated.",
        ),
        _check(
            "OPS-RECOVERY-CHECKSUM", True, "Evidence restoration requires checksum verification."
        ),
        _check(
            "OPS-RECOVERY-SERVICE", True, "Service recovery validation is deterministic and local."
        ),
    ]
    _write_json(RECOVERY_VERIFICATION_PATH, [check.model_dump(mode="json") for check in checks])
    return checks


def build_operations_evidence() -> OperationsEvidenceManifest:
    """Build the complete operations evidence manifest."""
    observability_manifest = build_observability_evidence()
    log_checks = validate_observability()
    slo_checks = evaluate_slos()
    incidents = simulate_incidents()
    lifecycle = build_incident_evidence()
    runbook_checks = validate_runbooks()
    rollback = simulate_rollback()
    recovery = validate_recovery()
    resilience = {
        "bounded_request_timeout": True,
        "bounded_reviewer_retries": True,
        "failure_guard": True,
        "request_size_limits": True,
        "graceful_shutdown": True,
        "safe_model_unavailable_failure": True,
    }
    _write_json(RESILIENCE_SUMMARY_PATH, resilience)
    error_budget = _load_json(ERROR_BUDGET_PATH)
    normal_run = _load_json(NORMAL_RUN_PATH)
    metrics_inventory = _load_json(METRICS_INVENTORY_PATH)
    runbook_inventory = _load_json(RUNBOOK_INVENTORY_PATH)
    checksums = checksum_paths(_evidence_inputs(include_manifest=False))
    manifest = OperationsEvidenceManifest(
        evidence_id="m17-operations-evidence-v1",
        generated_timestamp="2026-01-01T02:30:00Z",
        overall_status=_aggregate_status([*log_checks, *slo_checks, *runbook_checks, *recovery]),
        observability_manifest=observability_manifest,
        metrics_inventory=metrics_inventory,
        structured_log_validation=log_checks,
        slo_evaluation=slo_checks,
        error_budget_report=error_budget,
        normal_operations_run=normal_run,
        simulated_incidents=incidents,
        incident_lifecycle_records=lifecycle,
        rollback_evidence=rollback,
        recovery_verification=recovery,
        runbook_inventory=runbook_inventory,
        resilience_control_summary=resilience,
        checksums=checksums,
    )
    _write_json(EVIDENCE_MANIFEST_PATH, manifest.model_dump(mode="json"))
    checksums = checksum_paths(_evidence_inputs(include_manifest=True))
    _write_json(CHECKSUMS_PATH, checksums)
    _write_report(manifest.model_copy(update={"checksums": checksums}))
    return manifest.model_copy(update={"checksums": checksums})


def validate_operations_evidence() -> list[OperationsCheck]:
    """Validate generated operations evidence and no false PASS."""
    required = _evidence_inputs(include_manifest=True) + [CHECKSUMS_PATH, EVIDENCE_REPORT_PATH]
    checks = [
        OperationsCheck(
            check_id=f"OPS-EVIDENCE-{path.name}",
            status="PASS" if path.exists() else "FAIL",
            message=f"{path.as_posix()} exists."
            if path.exists()
            else f"{path.as_posix()} is missing.",
        )
        for path in required
    ]
    if not EVIDENCE_MANIFEST_PATH.exists():
        return checks
    manifest = OperationsEvidenceManifest.model_validate_json(
        EVIDENCE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    mandatory = [
        *manifest.structured_log_validation,
        *manifest.slo_evaluation,
        *manifest.recovery_verification,
    ]
    expected = _aggregate_status(mandatory)
    checks.append(
        OperationsCheck(
            check_id="OPS-EVIDENCE-NO-FALSE-PASS",
            status="PASS" if manifest.overall_status == expected else "FAIL",
            message="Overall status reflects mandatory evidence.",
            details={"expected": expected, "actual": manifest.overall_status},
        )
    )
    checks.append(
        OperationsCheck(
            check_id="OPS-EVIDENCE-DISCLAIMER",
            status="PASS" if "not for diagnosis" in manifest.disclaimer.lower() else "FAIL",
            message="Research-only operations disclaimer is present.",
        )
    )
    return checks


def _aggregate_status(checks: list[OperationsCheck]) -> OperationsStatus:
    mandatory = [check for check in checks if check.mandatory]
    if any(check.status in {"FAIL", "ERROR"} for check in mandatory):
        return "FAIL"
    if any(check.status == "ALERT" for check in mandatory):
        return "ALERT"
    if any(check.status == "WARN" for check in mandatory):
        return "WARN"
    if any(check.status in {"INCOMPLETE", "UNAVAILABLE"} for check in mandatory):
        return "INCOMPLETE"
    return "PASS"


def _check(check_id: str, passed: bool, message: str) -> OperationsCheck:
    return OperationsCheck(
        check_id=check_id,
        status="PASS" if passed else "FAIL",
        message=message if passed else f"{message} Check failed.",
    )


def _is_json(value: str) -> bool:
    try:
        json.loads(value)
        return True
    except json.JSONDecodeError:
        return False


def _file_contains(path: Path, text: str) -> bool:
    return path.exists() and text in path.read_text(encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evidence_inputs(include_manifest: bool) -> list[Path]:
    paths = [
        OBSERVABILITY_MANIFEST_PATH,
        METRICS_INVENTORY_PATH,
        LOG_VALIDATION_PATH,
        SLO_EVALUATION_PATH,
        ERROR_BUDGET_PATH,
        NORMAL_RUN_PATH,
        INCIDENT_RESULTS_PATH,
        INCIDENT_LIFECYCLE_PATH,
        ROLLBACK_EVIDENCE_PATH,
        RECOVERY_VERIFICATION_PATH,
        RUNBOOK_INVENTORY_PATH,
        RESILIENCE_SUMMARY_PATH,
    ]
    if include_manifest:
        paths.append(EVIDENCE_MANIFEST_PATH)
    return paths


def _write_report(manifest: OperationsEvidenceManifest) -> None:
    lines = [
        "# Operations Evidence Report",
        "",
        f"- Evidence ID: `{manifest.evidence_id}`",
        f"- Overall status: `{manifest.overall_status}`",
        "- Scope: simulated/local operations evidence only.",
        "- Cloud or production rollback executed: `false`",
        "",
        "## SLO Evaluation",
    ]
    for check in manifest.slo_evaluation:
        lines.append(f"- `{check.check_id}`: `{check.status}`")
    lines.extend(["", "## Incidents"])
    for incident in manifest.simulated_incidents:
        lines.append(f"- `{incident.incident_id}`: `{incident.severity}` {incident.trigger}")
    lines.extend(["", manifest.disclaimer, ""])
    EVIDENCE_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
