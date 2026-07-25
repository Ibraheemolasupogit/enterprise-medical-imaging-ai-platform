"""Pure formatting helpers for reviewer UI display and tests."""

from __future__ import annotations

from typing import Any

from medical_imaging_platform.reviewer_ui.models import ReviewerAPIError


def format_health_status(
    health: dict[str, Any] | None, ready: dict[str, Any] | None
) -> dict[str, str]:
    health_status = str((health or {}).get("status", "unavailable"))
    readiness = str((ready or {}).get("status", "not_ready"))
    return {
        "health": health_status,
        "readiness": readiness,
        "operational": "yes" if health_status == "healthy" and readiness == "ready" else "no",
    }


def format_segmentation_response(payload: dict[str, Any]) -> dict[str, object]:
    return {
        "status": payload.get("status", "unknown"),
        "predicted_voxel_count": int(payload.get("predicted_voxel_count", 0)),
        "predicted_volume_mm3": float(payload.get("predicted_volume_mm3", 0.0)),
        "probability_summary": payload.get("probability_summary", {}),
        "quality_status": payload.get("quality_status", "UNKNOWN"),
        "quality_findings": payload.get("quality_findings", []),
        "checkpoint_checksum": payload.get("checkpoint_checksum"),
        "output_paths": payload.get("output_paths", {}),
        "duration_ms": payload.get("duration_ms"),
        "disclaimer": payload.get("disclaimer"),
    }


def format_classification_response(payload: dict[str, Any]) -> dict[str, object]:
    label = str(payload.get("engineering_label", "indeterminate"))
    return {
        "raw_probability": payload.get("raw_probability"),
        "calibrated_probability": payload.get("calibrated_probability"),
        "threshold": payload.get("threshold"),
        "engineering_label": label,
        "abstained": bool(payload.get("abstained", label == "indeterminate")),
        "abstention_reason": payload.get("abstention_reason"),
        "quality_status": payload.get("quality_status", "UNKNOWN"),
        "checksums": {
            "checkpoint": payload.get("checkpoint_checksum"),
            "calibration": payload.get("calibration_checksum"),
            "threshold_policy": payload.get("threshold_policy_checksum"),
        },
        "duration_ms": payload.get("duration_ms"),
        "disclaimer": payload.get("disclaimer"),
    }


def format_longitudinal_response(payload: dict[str, Any]) -> dict[str, object]:
    labels = payload.get("engineering_label", [])
    if isinstance(labels, str):
        labels = [labels]
    return {
        "measurements": payload.get("measurements", {}),
        "match_summary": payload.get("match_summary", []),
        "change_metrics": payload.get("change_metrics", []),
        "engineering_label": labels,
        "indeterminate": "indeterminate" in labels,
        "upstream_quality_propagation": payload.get("upstream_quality_propagation", {}),
        "quality_findings": payload.get("quality_findings", []),
        "evidence_path": payload.get("evidence_path"),
        "duration_ms": payload.get("duration_ms"),
        "disclaimer": payload.get("disclaimer"),
    }


def format_evidence_response(payload: dict[str, Any]) -> dict[str, object]:
    return {
        "evidence_type": payload.get("evidence_type"),
        "status": payload.get("status"),
        "summary": payload.get("summary", {}),
        "quality_findings": payload.get("quality_findings", []),
        "provenance": payload.get("provenance", {}),
    }


def format_api_error(error: ReviewerAPIError) -> dict[str, object]:
    return error.to_display()
