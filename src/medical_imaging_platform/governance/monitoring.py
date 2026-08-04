"""Deterministic synthetic monitoring and drift evidence."""

from __future__ import annotations

import json
from pathlib import Path

from medical_imaging_platform.governance.models import (
    RESEARCH_DISCLAIMER,
    AlertSummary,
    CurrentMetric,
    DriftFinding,
    MetricBaseline,
    MonitoringBaseline,
    MonitoringRun,
    MonitoringStatus,
    MonitoringThresholds,
)
from medical_imaging_platform.release.checksums import checksum_paths

MONITORING_DIR = Path("reports/generated/monitoring")
BASELINE_PATH = MONITORING_DIR / "monitoring_baseline.json"
NORMAL_RUN_PATH = MONITORING_DIR / "monitoring_run_normal.json"
DRIFT_RUN_PATH = MONITORING_DIR / "monitoring_run_drift.json"
ALERT_SUMMARY_PATH = MONITORING_DIR / "alert_summary.json"
REPORT_PATH = MONITORING_DIR / "monitoring_evidence_report.md"
CHECKSUM_PATH = MONITORING_DIR / "checksum_manifest.json"


def default_thresholds() -> MonitoringThresholds:
    return MonitoringThresholds(warn_multiplier=2.0, alert_multiplier=3.0, alert_severity="high")


def build_baseline(output_path: Path = BASELINE_PATH) -> MonitoringBaseline:
    """Write deterministic synthetic monitoring baseline evidence."""
    thresholds = default_thresholds()
    specs = [
        ("request_volume", "operations", 100.0, 8.0),
        ("latency_ms_p95", "operations", 420.0, 35.0),
        ("inference_failure_rate", "operations", 0.01, 0.005),
        ("preprocessing_failure_rate", "input_quality", 0.02, 0.006),
        ("image_quality_failure_rate", "input_quality", 0.03, 0.008),
        ("segmentation_output_volume_ml", "segmentation_volume", 12.0, 2.0),
        ("segmentation_confidence_mean", "prediction_distribution", 0.76, 0.05),
        ("classification_probability_mean", "prediction_distribution", 0.58, 0.07),
        ("classification_abstention_rate", "classification_confidence", 0.10, 0.03),
        ("classification_confidence_mean", "classification_confidence", 0.81, 0.04),
        ("calibration_ece", "calibration", 0.05, 0.015),
    ]
    baseline = MonitoringBaseline(
        baseline_id="m14-synthetic-monitoring-baseline-v1",
        generated_timestamp="2026-01-01T00:00:00Z",
        window_label="synthetic-baseline-window",
        thresholds=thresholds,
        metrics=[
            MetricBaseline(
                metric_name=name,
                category=category,
                mean=mean,
                std=std,
                warn_delta=std * thresholds.warn_multiplier,
                alert_delta=std * thresholds.alert_multiplier,
            )
            for name, category, mean, std in specs
        ],
    )
    _write_json(output_path, baseline.model_dump(mode="json"))
    _write_report()
    _write_checksums()
    return baseline


def load_baseline(path: Path = BASELINE_PATH) -> MonitoringBaseline:
    if not path.is_file():
        return build_baseline(path)
    return MonitoringBaseline.model_validate_json(path.read_text(encoding="utf-8"))


def build_current_window(mode: str) -> list[CurrentMetric]:
    """Build deterministic normal or drifted synthetic current monitoring metrics."""
    baseline = load_baseline()
    drift_deltas = {
        "preprocessing_failure_rate": 0.03,
        "image_quality_failure_rate": 0.032,
        "segmentation_output_volume_ml": 7.0,
        "classification_confidence_mean": -0.14,
        "classification_probability_mean": 0.23,
        "calibration_ece": 0.055,
    }
    normal_deltas = {
        "request_volume": 4.0,
        "latency_ms_p95": 18.0,
        "inference_failure_rate": 0.002,
        "segmentation_confidence_mean": -0.02,
        "classification_abstention_rate": 0.01,
    }
    metrics: list[CurrentMetric] = []
    for item in baseline.metrics:
        delta = (
            drift_deltas.get(item.metric_name, 0.0)
            if mode == "simulated_drift"
            else normal_deltas.get(item.metric_name, 0.0)
        )
        metrics.append(
            CurrentMetric(
                metric_name=item.metric_name,
                category=item.category,
                value=round(item.mean + delta, 6),
                labels_available=item.category == "calibration",
            )
        )
    return metrics


def run_monitoring(
    *,
    mode: str = "normal",
    output_path: Path | None = None,
) -> MonitoringRun:
    """Compare current synthetic monitoring window with the stored baseline."""
    baseline = load_baseline()
    current = build_current_window(mode)
    baseline_by_name = {item.metric_name: item for item in baseline.metrics}
    findings = [_compare_metric(metric, baseline_by_name[metric.metric_name]) for metric in current]
    overall = _overall_status([finding.status for finding in findings])
    run = MonitoringRun(
        run_id=f"m14-monitoring-{mode}-v1",
        mode="simulated_drift" if mode == "simulated_drift" else "normal",
        generated_timestamp="2026-01-01T00:10:00Z" if mode == "normal" else "2026-01-01T00:20:00Z",
        overall_status=overall,
        metrics=current,
        findings=findings,
    )
    destination = output_path or (DRIFT_RUN_PATH if mode == "simulated_drift" else NORMAL_RUN_PATH)
    _write_json(destination, run.model_dump(mode="json"))
    if mode == "simulated_drift":
        build_alert_summary(run)
    _write_report()
    _write_checksums()
    return run


def build_alert_summary(run: MonitoringRun, path: Path = ALERT_SUMMARY_PATH) -> AlertSummary:
    """Write deterministic alert summary for a monitoring run."""
    alert_count = sum(1 for finding in run.findings if finding.status == "ALERT")
    warn_count = sum(1 for finding in run.findings if finding.status == "WARN")
    summary = AlertSummary(
        summary_id="m14-synthetic-alert-summary-v1",
        generated_timestamp="2026-01-01T00:21:00Z",
        overall_status=run.overall_status,
        alert_count=alert_count,
        warn_count=warn_count,
        recommended_action=(
            "Open human investigation, verify evidence integrity, review recent changes, and "
            "follow rollback/change-control documentation if engineering risk is confirmed. "
            "Do not infer clinical performance deterioration."
        ),
    )
    _write_json(path, summary.model_dump(mode="json"))
    return summary


def validate_monitoring_evidence(root: Path = MONITORING_DIR) -> list[dict[str, str]]:
    """Validate registry, monitoring, and audit evidence presence and boundaries."""
    required = [
        BASELINE_PATH,
        NORMAL_RUN_PATH,
        DRIFT_RUN_PATH,
        ALERT_SUMMARY_PATH,
        REPORT_PATH,
        CHECKSUM_PATH,
        Path("reports/generated/registry/registry_manifest.json"),
        Path("reports/generated/audit/audit_log.jsonl"),
    ]
    checks = []
    for path in required:
        checks.append(
            {
                "check_id": f"M14-EVIDENCE-{path.name.upper()}",
                "status": "PASS" if path.is_file() else "FAIL",
                "message": f"{path.as_posix()} is present.",
            }
        )
    text = ""
    for path in required:
        if path.is_file() and path.suffix in {".json", ".md", ".jsonl"}:
            text += path.read_text(encoding="utf-8")
    forbidden_claims = ("diagnostic performance deterioration", "clinical approval", "NHS approved")
    checks.append(
        {
            "check_id": "M14-EVIDENCE-NO-CLINICAL-CLAIMS",
            "status": "PASS" if not any(claim in text for claim in forbidden_claims) else "FAIL",
            "message": "Evidence avoids clinical approval and diagnostic-performance claims.",
        }
    )
    return checks


def _compare_metric(metric: CurrentMetric, baseline: MetricBaseline) -> DriftFinding:
    delta = abs(metric.value - baseline.mean)
    status: MonitoringStatus = "PASS"
    if delta > baseline.alert_delta:
        status = "ALERT"
    elif delta > baseline.warn_delta:
        status = "WARN"
    return DriftFinding(
        metric_name=metric.metric_name,
        category=metric.category,
        baseline_mean=baseline.mean,
        current_value=metric.value,
        absolute_delta=round(delta, 6),
        status=status,
        message=(
            f"{metric.metric_name} {status}; deterministic synthetic comparison only, "
            "not a clinical performance claim."
        ),
    )


def _overall_status(statuses: list[MonitoringStatus]) -> MonitoringStatus:
    if "ALERT" in statuses:
        return "ALERT"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report() -> None:
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Monitoring Evidence Report",
                "",
                RESEARCH_DISCLAIMER,
                "",
                "Synthetic metrics cover request volume, latency, inference failures, "
                "preprocessing failures, image-quality failures, segmentation volume and "
                "confidence, classification probability, abstention, confidence, and calibration.",
                "",
                "Drift checks are simple deterministic threshold comparisons and trigger human "
                "investigation only. They do not claim clinical performance deterioration.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_checksums() -> None:
    paths = [
        path
        for path in (
            BASELINE_PATH,
            NORMAL_RUN_PATH,
            DRIFT_RUN_PATH,
            ALERT_SUMMARY_PATH,
            REPORT_PATH,
        )
        if path.is_file()
    ]
    if paths:
        CHECKSUM_PATH.write_text(
            json.dumps(checksum_paths(paths), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
