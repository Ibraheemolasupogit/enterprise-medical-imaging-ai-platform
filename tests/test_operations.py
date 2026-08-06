import json
from pathlib import Path

import httpx
import yaml
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


def _write_operations_api_config(
    tmp_path: Path,
    *,
    segmentation_checkpoint: Path | None = None,
    classification_checkpoint: Path | None = None,
    classification_calibration: Path | None = None,
    classification_threshold_policy: Path | None = None,
    output_parent_exists: bool = True,
) -> Path:
    inputs = tmp_path / "inputs"
    evidence = tmp_path / "evidence"
    outputs = tmp_path / "outputs"
    for path in (inputs, evidence):
        path.mkdir(parents=True, exist_ok=True)
    if output_parent_exists:
        outputs.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "api": {
                "policy_version": "test-ops-api-v1",
                "service_name": "test-ops-api",
                "service_version": "0.1.0",
                "environment": "test",
                "host": "127.0.0.1",
                "port": 8001,
                "log_level": "INFO",
                "allowed_input_roots": [str(inputs), str(outputs)],
                "allowed_evidence_roots": [str(evidence), str(outputs)],
                "maximum_request_bytes": 1_000_000,
                "maximum_array_bytes": 2_000_000,
                "maximum_batch_size": 1,
                "request_timeout_seconds": 30,
                "enable_docs": False,
                "enable_openapi": False,
                "require_model_checksums": True,
                "require_quality_pass": True,
                "allow_degraded_review": True,
                "allow_external_bind": False,
                "allow_threshold_override": False,
                "segmentation_checkpoint": (
                    str(segmentation_checkpoint) if segmentation_checkpoint else None
                ),
                "classification_checkpoint": (
                    str(classification_checkpoint) if classification_checkpoint else None
                ),
                "classification_calibration": (
                    str(classification_calibration) if classification_calibration else None
                ),
                "classification_threshold_policy": (
                    str(classification_threshold_policy)
                    if classification_threshold_policy
                    else None
                ),
                "longitudinal_config": "config/longitudinal.yaml",
                "output_directory": str(outputs / "api"),
            }
        }
    }
    config_path = tmp_path / "api.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return config_path


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


def test_readiness_ready_when_runtime_dependencies_available(tmp_path: Path) -> None:
    artifacts = {
        "segmentation_checkpoint": tmp_path / "segmentation.pt",
        "classification_checkpoint": tmp_path / "classification.pt",
        "classification_calibration": tmp_path / "calibration.json",
        "classification_threshold_policy": tmp_path / "threshold_policy.json",
    }
    for path in artifacts.values():
        path.write_text("synthetic test artefact", encoding="utf-8")
    client = TestClient(create_app(_write_operations_api_config(tmp_path, **artifacts)))

    response = client.get("/ready")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ready"
    assert payload["degraded"] is False
    assert all(item["passed"] for item in payload["quality_findings"])


def test_readiness_degradation_for_missing_models(tmp_path: Path) -> None:
    client = TestClient(create_app(_write_operations_api_config(tmp_path)))

    response = client.get("/ready")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "degraded"
    assert payload["degraded"] is True
    assert any(
        item["rule_id"] == "API-QC-DEGRADED-001"
        and item["passed"] is True
        and item["severity"] == "WARN"
        for item in payload["quality_findings"]
    )


def test_readiness_not_ready_for_configured_missing_model(tmp_path: Path) -> None:
    missing_checkpoint = tmp_path / "missing-segmentation.pt"
    client = TestClient(
        create_app(
            _write_operations_api_config(tmp_path, segmentation_checkpoint=missing_checkpoint)
        )
    )

    response = client.get("/ready")
    payload = response.json()

    assert response.status_code == 503
    assert payload["error_code"] == "API-NOTREADY-503"
    assert payload["details"] == {"API-QC-MODEL-001": "segmentation_checkpoint exists."}


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
