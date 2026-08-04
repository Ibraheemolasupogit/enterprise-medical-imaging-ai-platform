import hashlib
import json
from pathlib import Path

import pytest

from medical_imaging_platform.cli import main
from medical_imaging_platform.governance.audit import (
    append_audit_event,
    build_audit_evidence,
    load_audit_log,
    output_checksum,
)
from medical_imaging_platform.governance.models import (
    ApprovalMetadata,
    AuditEvent,
    ModelVersionRecord,
    MonitoringThresholds,
)
from medical_imaging_platform.governance.monitoring import (
    build_baseline,
    load_baseline,
    run_monitoring,
    validate_monitoring_evidence,
)
from medical_imaging_platform.governance.registry import (
    approve_model,
    build_model_record,
    ensure_demo_registry,
    file_or_reference_checksum,
    load_registry,
    register_model,
    set_lifecycle_state,
)


def test_registry_lifecycle_duplicate_and_approval_rules(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry_manifest.json"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    record = build_model_record(
        model_type="segmentation",
        version="v1",
        checkpoint_reference=checkpoint.as_posix(),
        config_reference="config/segmentation.yaml",
        training_data_reference="synthetic://dataset",
    )

    manifest = register_model(record, registry_path)
    assert manifest.records[0].lifecycle_state == "candidate"
    assert (
        manifest.records[0].checkpoint_checksum
        == hashlib.sha256(b"synthetic checkpoint").hexdigest()
    )

    with pytest.raises(ValueError, match="Duplicate model version"):
        register_model(record, registry_path)

    approval = ApprovalMetadata(
        approved_by="reviewer",
        approval_ticket="M14-APPROVAL",
        approval_timestamp="2026-01-01T00:00:00Z",
        rationale="Synthetic engineering approval.",
    )
    approved = approve_model(record.model_name, record.version, approval, registry_path)

    assert approved.records[0].lifecycle_state == "approved"
    assert approved.records[0].approval_metadata == approval


def test_registry_reject_retire_and_missing_model_paths(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry_manifest.json"
    manifest = ensure_demo_registry(registry_path)
    record = manifest.records[0]

    rejected = set_lifecycle_state(record.model_name, record.version, "rejected", registry_path)
    assert rejected.records[0].lifecycle_state == "rejected"

    retired = set_lifecycle_state(record.model_name, record.version, "retired", registry_path)
    assert retired.records[0].lifecycle_state == "retired"

    with pytest.raises(ValueError, match="Use approve_model"):
        set_lifecycle_state(record.model_name, record.version, "approved", registry_path)

    with pytest.raises(ValueError, match="Model version not found"):
        set_lifecycle_state("missing", "v0", "rejected", registry_path)

    approval = ApprovalMetadata(
        approved_by="reviewer",
        approval_ticket="M14-MISSING",
        approval_timestamp="2026-01-01T00:00:00Z",
        rationale="Synthetic engineering approval.",
    )
    with pytest.raises(ValueError, match="Model version not found"):
        approve_model("missing", "v0", approval, registry_path)


def test_approved_model_requires_metadata_and_checksum_validation() -> None:
    checksum = file_or_reference_checksum("synthetic://checkpoint")
    assert checksum == file_or_reference_checksum("synthetic://checkpoint")

    with pytest.raises(ValueError, match="approval metadata"):
        ModelVersionRecord(
            model_name="synthetic-segmentation-baseline",
            version="v1",
            model_type="segmentation",
            lifecycle_state="approved",
            checkpoint_reference="synthetic://checkpoint",
            checkpoint_checksum=checksum,
            framework_versions={"torch": "2.13.0+cpu"},
            config_reference="config/segmentation.yaml",
            config_checksum=file_or_reference_checksum("config/segmentation.yaml"),
            training_data_reference="synthetic://dataset",
        )


def test_monitoring_baseline_normal_and_drift(tmp_path: Path) -> None:
    baseline = build_baseline(tmp_path / "baseline.json")

    assert baseline.baseline_id == "m14-synthetic-monitoring-baseline-v1"
    assert {metric.metric_name for metric in baseline.metrics} >= {
        "request_volume",
        "latency_ms_p95",
        "segmentation_output_volume_ml",
        "classification_abstention_rate",
        "calibration_ece",
    }

    normal = run_monitoring(mode="normal", output_path=tmp_path / "normal.json")
    drift = run_monitoring(mode="simulated_drift", output_path=tmp_path / "drift.json")

    assert normal.overall_status == "PASS"
    assert drift.overall_status in {"WARN", "ALERT"}
    assert any(finding.status == "ALERT" for finding in drift.findings)
    assert all(
        "clinical performance deterioration detected" not in finding.message
        for finding in drift.findings
    )


def test_monitoring_threshold_and_missing_baseline_branches(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Alert multiplier"):
        MonitoringThresholds(warn_multiplier=3.0, alert_multiplier=2.0, alert_severity="high")

    missing_baseline = tmp_path / "missing-baseline.json"
    assert not missing_baseline.exists()
    baseline = load_baseline(missing_baseline)

    assert missing_baseline.is_file()
    assert baseline.thresholds.alert_multiplier == 3.0


def test_audit_append_and_sensitive_field_rejection(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    event = AuditEvent(
        request_id="req-1",
        correlation_id="corr-1",
        timestamp="2026-01-01T00:00:00Z",
        model_version="v1",
        config_version="config/segmentation.yaml",
        input_provenance_reference="synthetic://case-1",
        output_checksum=output_checksum("result"),
        reviewer_action="accepted",
        actor_type="human_reviewer",
    )
    append_audit_event(event, audit_path)
    append_audit_event(event.model_copy(update={"request_id": "req-2"}), audit_path)

    loaded = load_audit_log(audit_path)
    assert [item.request_id for item in loaded] == ["req-1", "req-2"]
    assert load_audit_log(tmp_path / "missing.jsonl") == []

    with pytest.raises(ValueError, match="PHI-sensitive"):
        AuditEvent(
            request_id="req-phi",
            correlation_id="corr-phi",
            timestamp="2026-01-01T00:00:00Z",
            model_version="v1",
            config_version="config/segmentation.yaml",
            input_provenance_reference="synthetic://case-1",
            output_checksum=output_checksum("result"),
            actor_type="system",
            metadata={"patient_id": "forbidden"},
        )


def test_evidence_validation_and_cli_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["register-model"]) == 0
    assert "model records" in capsys.readouterr().out

    assert main(["list-models"]) == 0
    assert "Registry contains" in capsys.readouterr().out

    assert (
        main(
            [
                "approve-model",
                "--model-name",
                "synthetic-segmentation-baseline",
                "--version",
                "m14-segmentation-synthetic-v1",
                "--approved-by",
                "reviewer",
                "--approval-ticket",
                "M14-CLI",
                "--rationale",
                "Synthetic approval.",
            ]
        )
        == 0
    )
    assert "Approved synthetic-segmentation-baseline" in capsys.readouterr().out

    assert main(["build-monitoring-baseline"]) == 0
    assert main(["run-monitoring"]) == 0
    assert main(["simulate-monitoring-drift"]) == 0
    assert main(["build-audit-evidence"]) == 0
    assert main(["validate-monitoring-evidence"]) == 0

    validation = validate_monitoring_evidence()
    assert all(check["status"] == "PASS" for check in validation)

    registry = load_registry()
    assert any(record.model_type == "classification" for record in registry.records)

    events = build_audit_evidence()
    assert len(events) == 2


def test_monitoring_evidence_contains_no_clinical_claims() -> None:
    build_baseline()
    run_monitoring(mode="normal")
    run_monitoring(mode="simulated_drift")
    build_audit_evidence()
    checks = validate_monitoring_evidence()

    assert any(check["check_id"] == "M14-EVIDENCE-NO-CLINICAL-CLAIMS" for check in checks)
    assert all(check["status"] == "PASS" for check in checks)
    report = Path("reports/generated/monitoring/monitoring_evidence_report.md").read_text(
        encoding="utf-8"
    )
    assert "do not claim clinical performance deterioration" in report
    assert "NHS approval" in report

    payload = json.loads(Path("reports/generated/monitoring/monitoring_run_drift.json").read_text())
    assert payload["overall_status"] == "ALERT"
