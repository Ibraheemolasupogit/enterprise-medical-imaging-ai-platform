import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

import medical_imaging_platform.operations.assurance as ops
from medical_imaging_platform.api.app import create_app
from medical_imaging_platform.api.observability import (
    MetricsRegistry,
    metric_labels_safe,
    redact,
    structured_log_event,
)
from medical_imaging_platform.cli import main
from medical_imaging_platform.operations.assurance import (
    build_operations_evidence,
    evaluate_slos,
    simulate_incidents,
    simulate_rollback,
    transition_incident,
    validate_observability,
    validate_operations_evidence,
    validate_recovery,
    validate_runbooks,
)
from medical_imaging_platform.reviewer_ui.api_client import ReviewerAPIClient
from medical_imaging_platform.reviewer_ui.config import load_reviewer_ui_config
from medical_imaging_platform.reviewer_ui.models import ReviewerAPIError


def _redirect_operations_evidence(monkeypatch, tmp_path: Path) -> Path:  # type: ignore[no-untyped-def]
    evidence_dir = tmp_path / "operations"
    names = [
        "OBSERVABILITY_MANIFEST_PATH",
        "METRICS_INVENTORY_PATH",
        "LOG_VALIDATION_PATH",
        "SLO_EVALUATION_PATH",
        "ERROR_BUDGET_PATH",
        "NORMAL_RUN_PATH",
        "INCIDENT_RESULTS_PATH",
        "INCIDENT_LIFECYCLE_PATH",
        "ROLLBACK_EVIDENCE_PATH",
        "RECOVERY_VERIFICATION_PATH",
        "RUNBOOK_INVENTORY_PATH",
        "RESILIENCE_SUMMARY_PATH",
        "EVIDENCE_MANIFEST_PATH",
        "EVIDENCE_REPORT_PATH",
        "CHECKSUMS_PATH",
    ]
    monkeypatch.setattr(ops, "OPERATIONS_EVIDENCE_DIR", evidence_dir)
    for name in names:
        current = getattr(ops, name)
        monkeypatch.setattr(ops, name, evidence_dir / current.name)
    return evidence_dir


def test_metrics_registry_prometheus_format_and_bounded_labels() -> None:
    registry = MetricsRegistry()
    registry.record_request("GET", "/health", 200, 0.01)
    registry.record_request("POST", "/sensitive/patient/123/file.dcm", 500, 0.2)
    registry.record_inference("segmentation", "success")
    registry.record_inference("patient_id", "secret")
    registry.set_readiness(ready=False, degraded=True)

    metrics = registry.render_prometheus()

    assert "medical_imaging_api_requests_total" in metrics
    assert 'route="/other"' in metrics
    assert "patient" not in metrics.lower()
    assert "secret" not in metrics.lower()
    assert metric_labels_safe(metrics)


def test_metrics_endpoint_disabled_and_protected(tmp_path: Path) -> None:
    config_path = tmp_path / "api.yaml"
    config_path.write_text(
        Path("config/api.yaml")
        .read_text(encoding="utf-8")
        .replace("enable_metrics_endpoint: false", "enable_metrics_endpoint: true")
        .replace("metrics_access_token: null", 'metrics_access_token: "ops-token"'),
        encoding="utf-8",
    )
    app = create_app(config_path)
    client = TestClient(app)

    assert client.get("/metrics").status_code == 403
    response = client.get("/metrics", headers={"X-Metrics-Token": "ops-token"})
    assert response.status_code == 200
    assert "medical_imaging_api_readiness" in response.text
    assert metric_labels_safe(response.text)


def test_default_metrics_endpoint_disabled() -> None:
    client = TestClient(create_app())

    assert client.get("/metrics").status_code == 404


def test_structured_logging_redaction_and_correlation_ids() -> None:
    event = structured_log_event(
        service="api",
        event_type="audit_event",
        request_id="req-1",
        correlation_id="corr-1",
        model_version="m17-synthetic",
        provenance_ref="synthetic-case-001",
        details={"patient_id": "123", "token": "Bearer secret", "safe": "ok"},
    )
    payload = json.loads(event)

    assert payload["request_id"] == "req-1"
    assert payload["correlation_id"] == "corr-1"
    assert payload["details"]["patient_id"] == "[REDACTED]"
    assert "secret" not in event
    assert redact({"raw_payload": [1, 2, 3]})["raw_payload"] == "[REDACTED]"


def test_readiness_degradation_for_missing_models() -> None:
    client = TestClient(create_app())

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["degraded"] is True


class TimeoutTransport(httpx.BaseTransport):
    def __init__(self) -> None:
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        raise httpx.TimeoutException("timeout", request=request)


def test_reviewer_ui_retry_exhaustion_and_circuit_breaker() -> None:
    config = load_reviewer_ui_config(Path("config/reviewer_ui.yaml"))
    transport = TimeoutTransport()
    client = ReviewerAPIClient(
        config,
        client=httpx.Client(transport=transport, base_url=config.api_base_url),
        max_retries=1,
        failure_threshold=2,
    )

    try:
        client.health("req-1")
    except ReviewerAPIError as exc:
        assert exc.error_code == "UI-API-UNAVAILABLE"
    else:
        raise AssertionError("Expected retry exhaustion.")

    try:
        client.health("req-2")
    except ReviewerAPIError as exc:
        assert exc.error_code == "UI-API-CIRCUIT-OPEN"
    else:
        raise AssertionError("Expected circuit breaker.")
    assert transport.calls == 2


def test_slos_error_budget_and_incident_simulation(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _redirect_operations_evidence(monkeypatch, tmp_path)
    slos = evaluate_slos()
    incidents = simulate_incidents()

    assert all(check.status == "PASS" for check in slos)
    assert len(incidents) == 10
    assert {incident.status for incident in incidents} == {"post-incident-review-required"}
    assert all("clinical" not in item.detection_evidence.lower() for item in incidents)


def test_incident_lifecycle_transitions_and_invalid_rejection() -> None:
    record = transition_incident("INC-001", "detected", "acknowledged", "actor")

    assert record["actor"] == "actor"
    assert record["automatic_closure"] is False
    try:
        transition_incident("INC-001", "detected", "closed", "actor")
    except ValueError as exc:
        assert "Invalid incident transition" in str(exc)
    else:
        raise AssertionError("Expected invalid transition rejection.")


def test_rollback_requires_approval_and_recovery_verifies(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _redirect_operations_evidence(monkeypatch, tmp_path)
    rollback = simulate_rollback()
    recovery = validate_recovery()

    assert rollback["operator_approval"]["approval_ticket"]
    assert rollback["automated_production_rollback"] is False
    assert all(check.status == "PASS" for check in recovery)


def test_runbook_completeness() -> None:
    checks = validate_runbooks()

    assert len(checks) == 12
    assert all(check.status == "PASS" for check in checks)


def test_operations_evidence_deterministic_and_no_false_pass(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _redirect_operations_evidence(monkeypatch, tmp_path)
    first = build_operations_evidence()
    second = build_operations_evidence()
    checks = validate_operations_evidence()

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.overall_status == "PASS"
    assert all(check.status == "PASS" for check in checks)


def test_operations_evidence_missing_paths_fail(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _redirect_operations_evidence(monkeypatch, tmp_path)
    checks = validate_operations_evidence()

    assert any(check.status == "FAIL" for check in checks)


def test_validate_observability_and_cli_commands(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    _redirect_operations_evidence(monkeypatch, tmp_path)

    assert all(check.status == "PASS" for check in validate_observability())
    for command in [
        "validate-observability",
        "build-observability-evidence",
        "evaluate-slos",
        "simulate-incidents",
        "build-incident-evidence",
        "validate-runbooks",
        "simulate-rollback",
        "validate-recovery",
        "build-operations-evidence",
        "validate-operations-evidence",
    ]:
        assert main([command]) == 0
        assert capsys.readouterr().out
