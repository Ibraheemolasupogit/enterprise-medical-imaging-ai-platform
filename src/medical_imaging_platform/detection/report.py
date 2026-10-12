"""Markdown reports for detection evidence."""

from __future__ import annotations

from typing import Any


def render_detection_report(payload: dict[str, Any]) -> str:
    test_metrics = payload["evaluation"]["test"]["metrics"]
    lines = [
        "# Synthetic 2D Lesion Detection Evidence",
        "",
        "Research and engineering evidence only. This detector is not for diagnosis, triage, "
        "RECIST assessment, patient management, or medical-device use.",
        "",
        f"- Experiment ID: `{payload['experiment_id']}`",
        f"- Dataset ID: `{payload['dataset_id']}`",
        f"- Model family: `{payload['model_family']}`",
        f"- Status: `{payload['status']}`",
        f"- Random seed: `{payload['random_seed']}`",
        f"- Parameters: `{payload['model_parameter_count']}`",
        "",
        "## Coordinate Convention",
        "2D detections use half-open `xyxy` pixel boxes on axial slices. `source_z` links each "
        "slice back to the canonical `z,y,x` synthetic volume.",
        "",
        "## Test Metrics",
        f"- AP@0.50: `{test_metrics.get('ap50')}`",
        f"- AP@0.75: `{test_metrics.get('ap75')}`",
        f"- mAP: `{test_metrics.get('map')}`",
        f"- Precision: `{test_metrics.get('precision')}`",
        f"- Recall: `{test_metrics.get('recall')}`",
        f"- False positives per image: `{test_metrics.get('false_positives_per_image')}`",
        "",
        "## Limitations",
    ]
    lines.extend(f"- {item}" for item in payload.get("limitations", []))
    lines.append("")
    return "\n".join(lines)


def render_benchmark_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Detection Faster R-CNN Vs Tiny DETR Benchmark",
        "",
        "Synthetic/de-identified engineering benchmark only. No clinical, diagnostic, triage, "
        "RECIST, patient-management, or medical-device claim is made.",
        "",
        f"- Benchmark ID: `{payload['benchmark_id']}`",
        f"- Dataset ID: `{payload['dataset_id']}`",
        f"- Random seed: `{payload['fairness_attestation']['random_seed']}`",
        "",
        "## Fairness Controls",
    ]
    fairness = payload["fairness_attestation"]
    lines.extend(
        [
            f"- Manifest checksum: `{fairness['detection_manifest_checksum']}`",
            f"- Split counts: `{fairness['split_counts']}`",
            f"- Positive/negative counts: `{fairness['positive_negative_counts']}`",
            f"- {fairness['shared_dataset_statement']}",
            f"- {fairness['shared_training_policy_statement']}",
            f"- {fairness['shared_metric_statement']}",
            "",
            "## Model Results",
        ]
    )
    for family, result in payload["models"].items():
        metrics = result["test_metrics"]
        lines.extend(
            [
                f"### {family}",
                f"- Status: `{result['status']}`",
                f"- Parameters: `{result['parameter_count']}`",
                f"- Checkpoint size bytes: `{result['checkpoint_size_bytes']}`",
                f"- AP@0.50: `{metrics.get('ap50')}`",
                f"- AP@0.75: `{metrics.get('ap75')}`",
                f"- mAP: `{metrics.get('map')}`",
                f"- Latency mean seconds: `{result['latency_summary']['mean_seconds']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Limits",
            "- Results are specific to this synthetic benchmark configuration.",
            "- The comparison does not declare a universally better detector.",
            "- Detector confidence is not a clinical probability.",
            "",
        ]
    )
    return "\n".join(lines)
