"""Safe local observability helpers for the governed API."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

SENSITIVE_KEYS = {
    "patient_name",
    "patient_id",
    "nhs_number",
    "dicom",
    "pixel_array",
    "image",
    "array",
    "payload",
    "token",
    "secret",
    "password",
    "authorization",
    "cookie",
    "filename",
    "path",
}
ALLOWED_LABELS = {
    "method",
    "route",
    "status_class",
    "outcome",
    "readiness",
    "model_type",
    "service",
}


@dataclass
class MetricsRegistry:
    """Small bounded Prometheus-compatible registry with fixed label keys."""

    request_count: dict[tuple[str, str, str], int] = field(default_factory=dict)
    latency_seconds: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    inference_outcomes: dict[tuple[str, str], int] = field(default_factory=dict)
    readiness: dict[str, int] = field(default_factory=lambda: {"ready": 0, "degraded": 0})
    model_version: str = "unconfigured"

    def record_request(self, method: str, route: str, status_code: int, latency: float) -> None:
        safe_route = _safe_route(route)
        status_class = f"{status_code // 100}xx"
        self.request_count[(method.upper(), safe_route, status_class)] = (
            self.request_count.get((method.upper(), safe_route, status_class), 0) + 1
        )
        samples = self.latency_seconds.setdefault((method.upper(), safe_route), [])
        samples.append(max(0.0, latency))
        del samples[:-100]

    def record_inference(self, model_type: str, outcome: str) -> None:
        safe_model_type = _safe_label(
            model_type, {"segmentation", "classification", "longitudinal"}
        )
        safe_outcome = _safe_label(outcome, {"success", "failure", "abstained", "degraded"})
        self.inference_outcomes[(safe_model_type, safe_outcome)] = (
            self.inference_outcomes.get((safe_model_type, safe_outcome), 0) + 1
        )

    def set_readiness(self, ready: bool, degraded: bool = False) -> None:
        self.readiness["ready"] = 1 if ready else 0
        self.readiness["degraded"] = 1 if degraded else 0

    def render_prometheus(self) -> str:
        lines = [
            "# HELP medical_imaging_api_requests_total Total bounded API requests.",
            "# TYPE medical_imaging_api_requests_total counter",
        ]
        for (method, route, status_class), value in sorted(self.request_count.items()):
            lines.append(
                "medical_imaging_api_requests_total"
                f'{{method="{method}",route="{route}",status_class="{status_class}"}} {value}'
            )
        lines.extend(
            [
                "# HELP medical_imaging_api_request_latency_seconds Average API request latency.",
                "# TYPE medical_imaging_api_request_latency_seconds gauge",
            ]
        )
        for (method, route), samples in sorted(self.latency_seconds.items()):
            average = sum(samples) / len(samples)
            lines.append(
                "medical_imaging_api_request_latency_seconds"
                f'{{method="{method}",route="{route}"}} {average:.6f}'
            )
        lines.extend(
            [
                "# HELP medical_imaging_inference_outcomes_total Inference outcome counts.",
                "# TYPE medical_imaging_inference_outcomes_total counter",
            ]
        )
        for (model_type, outcome), value in sorted(self.inference_outcomes.items()):
            lines.append(
                "medical_imaging_inference_outcomes_total"
                f'{{model_type="{model_type}",outcome="{outcome}"}} {value}'
            )
        lines.extend(
            [
                "# HELP medical_imaging_api_readiness API readiness state.",
                "# TYPE medical_imaging_api_readiness gauge",
                f'medical_imaging_api_readiness{{readiness="ready"}} {self.readiness["ready"]}',
                "medical_imaging_api_readiness"
                f'{{readiness="degraded"}} {self.readiness["degraded"]}',
                "# HELP medical_imaging_active_model_version Active configured model version.",
                "# TYPE medical_imaging_active_model_version gauge",
                'medical_imaging_active_model_version{model_type="configured"} 1',
            ]
        )
        return "\n".join(lines) + "\n"


def structured_log_event(
    *,
    service: str,
    event_type: str,
    severity: str = "INFO",
    request_id: str | None = None,
    correlation_id: str | None = None,
    model_version: str | None = None,
    provenance_ref: str | None = None,
    details: dict[str, Any] | None = None,
) -> str:
    """Build a redacted structured JSON log event."""
    event = {
        "timestamp": "2026-01-01T00:50:00Z",
        "severity": severity,
        "service": service,
        "event_type": event_type,
        "request_id": request_id or "not-set",
        "correlation_id": correlation_id or request_id or "not-set",
    }
    if model_version is not None:
        event["model_version"] = model_version
    if provenance_ref is not None:
        event["provenance_ref"] = provenance_ref
    event["details"] = redact(details or {})
    return json.dumps(event, sort_keys=True)


def redact(value: Any) -> Any:
    """Redact sensitive fields recursively."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in SENSITIVE_KEYS):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = redact(item)
        return redacted
    if isinstance(value, list):
        return ["[REDACTED_LIST]" if len(value) > 8 else redact(item) for item in value[:8]]
    if isinstance(value, str) and ("Bearer " in value or "AKIA" in value):
        return "[REDACTED]"
    return value


def metric_labels_safe(metrics_text: str) -> bool:
    """Reject sensitive or arbitrary metric labels."""
    for line in metrics_text.splitlines():
        if "{" not in line or "}" not in line:
            continue
        label_text = line.split("{", 1)[1].split("}", 1)[0]
        for label in label_text.split(","):
            key = label.split("=", 1)[0].strip()
            if key and key not in ALLOWED_LABELS:
                return False
            if any(token in label.lower() for token in SENSITIVE_KEYS):
                return False
    return True


def _safe_route(route: str) -> str:
    known = {
        "/health",
        "/ready",
        "/version",
        "/metrics",
        "/v1/segmentation/predict",
        "/v1/classification/predict",
        "/v1/longitudinal/analyse",
    }
    return route if route in known else "/other"


def _safe_label(value: str, allowed: set[str]) -> str:
    return value if value in allowed else "other"


def now_seconds() -> float:
    return time.perf_counter()
