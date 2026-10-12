"""Common Faster R-CNN versus Tiny DETR detection benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_imaging_platform.detection.dataset import validate_detection_dataset
from medical_imaging_platform.detection.models import DetectionConfig, DetectionModelFamily
from medical_imaging_platform.detection.pipeline import (
    configuration_checksum,
    manifest_checksum,
    safe_output_dir,
)
from medical_imaging_platform.detection.report import render_benchmark_report
from medical_imaging_platform.detection.trainer import train_detection_experiment
from medical_imaging_platform.synthetic.manifest import sha256_file, stable_timestamp

BENCHMARK_FAMILIES: tuple[DetectionModelFamily, DetectionModelFamily] = ("faster_rcnn", "tiny_detr")


def run_detection_benchmark(
    dataset_dir: Path,
    *,
    output_root: Path,
    config: DetectionConfig,
    overwrite: bool,
) -> dict[str, Any]:
    """Train both detection families with identical dataset and evaluation policy."""
    manifest = validate_detection_dataset(dataset_dir)
    benchmark_id = (
        f"detection-benchmark-{manifest.dataset_id}-{config.policy_version}-"
        f"seed-{config.random_seed}"
    )
    output_dir = safe_output_dir(output_root, benchmark_id)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Detection benchmark already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_root = output_dir / "experiments"
    source_manifest_checksum = manifest_checksum(dataset_dir / "detection_dataset_manifest.json")
    model_results: dict[str, dict[str, Any]] = {}
    for family in BENCHMARK_FAMILIES:
        family_config = config.model_copy(update={"model_family": family})
        payload = train_detection_experiment(
            dataset_dir,
            output_root=experiments_root,
            config=family_config,
            overwrite=overwrite,
        )
        experiment_dir = experiments_root / payload["experiment_id"]
        best_model = experiment_dir / payload["paths"]["best_model"]
        metrics = payload["evaluation"]["test"]["metrics"]
        model_results[family] = {
            "experiment_id": payload["experiment_id"],
            "experiment_dir": str(experiment_dir.relative_to(output_dir)),
            "model_family": family,
            "architecture": payload["model_architecture"],
            "status": payload["status"],
            "quality_findings": payload["quality_findings"],
            "validation_metrics": payload["evaluation"]["validation"]["metrics"],
            "test_metrics": metrics,
            "parameter_count": payload["model_parameter_count"],
            "checkpoint_size_bytes": best_model.stat().st_size,
            "checkpoint_checksum": sha256_file(best_model),
            "training_duration_seconds": payload["training_duration_seconds"],
            "latency_summary": metrics["latency_summary"],
            "dataset_manifest_checksum": payload["dataset_manifest_checksum"],
            "configuration_checksum": payload["configuration_checksum"],
            "random_seed": payload["random_seed"],
        }
    fairness = {
        "dataset_id": manifest.dataset_id,
        "detection_manifest_checksum": source_manifest_checksum,
        "split_counts": manifest.split_counts,
        "positive_negative_counts": {
            "positive": manifest.positive_image_count,
            "negative": manifest.negative_image_count,
            "boxes": manifest.box_count,
        },
        "config_checksum": configuration_checksum(config),
        "random_seed": config.random_seed,
        "shared_dataset_statement": (
            "Both models used the exact same prepared detection_dataset_manifest.json, slice "
            "images, boxes, labels, subject splits, source checksums, and coordinate convention."
        ),
        "shared_training_policy_statement": (
            "Both models used the same epochs, optimizer, learning rate, batch size, seed, score "
            "threshold, NMS threshold, and evaluation IoU thresholds."
        ),
        "shared_metric_statement": "Both models used the same detection metric implementation.",
    }
    comparison = _comparison(model_results)
    payload = {
        "benchmark_id": benchmark_id,
        "generated_at": stable_timestamp(),
        "policy_version": config.policy_version,
        "dataset_id": manifest.dataset_id,
        "fairness_attestation": fairness,
        "models": model_results,
        "comparison": comparison,
        "limitations": [
            "Synthetic/de-identified engineering benchmark configuration only.",
            "No clinical-performance, diagnostic, triage, RECIST, patient-management, "
            "or medical-device claim.",
            "Comparison does not declare a universally better detector.",
        ],
        "output_paths": {
            "manifest": "detection_benchmark_manifest.json",
            "comparison": "detection_benchmark_comparison.json",
            "report": "detection_benchmark_report.md",
            "checksum_manifest": "evidence/checksum_manifest.json",
        },
    }
    _write_json(output_dir / "detection_benchmark_manifest.json", payload)
    _write_json(
        output_dir / "detection_benchmark_comparison.json",
        {
            "benchmark_id": benchmark_id,
            "models": model_results,
            "comparison": comparison,
            "fairness_attestation": fairness,
            "limitations": payload["limitations"],
        },
    )
    _write_text(output_dir / "detection_benchmark_report.md", render_benchmark_report(payload))
    checksums = _checksum_manifest(output_dir)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _write_json(evidence_dir / "checksum_manifest.json", checksums)
    payload["checksums"] = checksums
    _write_json(output_dir / "detection_benchmark_manifest.json", payload)
    return payload


def _comparison(model_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    faster = model_results["faster_rcnn"]
    detr = model_results["tiny_detr"]
    metric_deltas = {
        key: _delta(faster["test_metrics"].get(key), detr["test_metrics"].get(key))
        for key in ("ap50", "ap75", "map", "precision", "recall", "false_positives_per_image")
    }
    return {
        "delta_direction": "tiny_detr_minus_faster_rcnn",
        "test_metric_deltas": metric_deltas,
        "parameter_count_delta": detr["parameter_count"] - faster["parameter_count"],
        "checkpoint_size_bytes_delta": detr["checkpoint_size_bytes"]
        - faster["checkpoint_size_bytes"],
        "training_duration_seconds_delta": detr["training_duration_seconds"]
        - faster["training_duration_seconds"],
        "latency_mean_seconds_delta": detr["latency_summary"]["mean_seconds"]
        - faster["latency_summary"]["mean_seconds"],
        "interpretation": (
            "Deltas are specific to this synthetic/de-identified engineering benchmark and do not "
            "establish a universally better detector."
        ),
    }


def _delta(left: object, right: object) -> float | None:
    if not isinstance(left, int | float) or not isinstance(right, int | float):
        return None
    return float(right - left)


def _checksum_manifest(output_dir: Path) -> dict[str, Any]:
    checksums = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "checksum_manifest.json":
            continue
        checksums[str(path.relative_to(output_dir))] = sha256_file(path)
    return {"generated_at": stable_timestamp(), "algorithm": "sha256", "files": checksums}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
