"""Common CNN versus ViT classification benchmark runner."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from medical_imaging_platform.classification.dataset import validate_classification_dataset
from medical_imaging_platform.classification.inference import classify_array
from medical_imaging_platform.classification.models import ClassificationConfig, ModelFamily
from medical_imaging_platform.classification.pipeline import (
    configuration_checksum,
    manifest_checksum,
    safe_output_dir,
)
from medical_imaging_platform.classification.trainer import train_classification_experiment
from medical_imaging_platform.synthetic.manifest import sha256_file, stable_timestamp

BENCHMARK_FAMILIES: tuple[ModelFamily, ModelFamily] = ("small_3d_cnn", "tiny_3d_vit")


def run_classification_benchmark(
    dataset_dir: Path,
    *,
    output_root: Path,
    config: ClassificationConfig,
    overwrite: bool,
) -> dict[str, Any]:
    """Train both supported classifier families with a shared dataset and policy."""
    manifest = validate_classification_dataset(dataset_dir)
    benchmark_id = (
        f"classification-benchmark-{manifest.dataset_id}-{config.policy_version}-"
        f"seed-{config.random_seed}"
    )
    output_dir = safe_output_dir(output_root, benchmark_id)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Classification benchmark already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_root = output_dir / "experiments"

    source_manifest_checksum = manifest_checksum(dataset_dir / "classification_manifest.json")
    model_results: dict[str, dict[str, Any]] = {}
    for family in BENCHMARK_FAMILIES:
        family_config = config.model_copy(update={"model_family": family})
        payload = train_classification_experiment(
            dataset_dir,
            output_root=experiments_root,
            config=family_config,
            overwrite=overwrite,
        )
        experiment_dir = experiments_root / payload["experiment_id"]
        latency = _latency_summary(dataset_dir, manifest, experiment_dir, family_config)
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
            "latency_summary": latency,
            "dataset_manifest_checksum": payload["dataset_manifest_checksum"],
            "configuration_checksum": payload["configuration_checksum"],
            "random_seed": payload["random_seed"],
        }

    fairness = {
        "dataset_id": manifest.dataset_id,
        "classification_manifest_checksum": source_manifest_checksum,
        "split_counts": manifest.split_counts,
        "config_checksum": configuration_checksum(config),
        "random_seed": config.random_seed,
        "shared_preprocessing_statement": (
            "Both models used the same prepared classification image.npy arrays; no dataset "
            "preprocessing or split generation was rerun inside the benchmark."
        ),
        "shared_transforms_statement": (
            "Both models used the existing classification transform builder with the same split "
            "names, seed, and training policy."
        ),
        "shared_labels_statement": (
            "Both models used identical binary synthetic lesion-presence labels from the same "
            "classification manifest."
        ),
        "shared_optimizer_training_policy_statement": (
            "Both models used the same epochs, optimizer, learning rate, weight decay, batch size, "
            "calibration method, threshold-selection method, and quality gates."
        ),
    }
    comparison = _comparison(model_results)
    payload = {
        "benchmark_id": benchmark_id,
        "generated_at": stable_timestamp(),
        "policy_version": config.policy_version,
        "dataset_id": manifest.dataset_id,
        "benchmark_fairness_attestation": fairness,
        "models": model_results,
        "comparison": comparison,
        "limitations": [
            "Synthetic or de-identified engineering benchmark configuration only.",
            "No clinical-performance, diagnostic, or medical-device claim is made.",
            "Comparison does not declare a universally better model.",
        ],
        "output_paths": {
            "manifest": "classification_benchmark_manifest.json",
            "comparison": "classification_benchmark_comparison.json",
            "report": "classification_benchmark_report.md",
            "checksum_manifest": "evidence/checksum_manifest.json",
        },
    }
    _write_json(output_dir / "classification_benchmark_manifest.json", payload)
    _write_json(
        output_dir / "classification_benchmark_comparison.json",
        {
            "benchmark_id": benchmark_id,
            "models": model_results,
            "comparison": comparison,
            "benchmark_fairness_attestation": fairness,
            "limitations": payload["limitations"],
        },
    )
    _write_text(output_dir / "classification_benchmark_report.md", _render_report(payload))
    checksum_manifest = _checksum_manifest(output_dir)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _write_json(evidence_dir / "checksum_manifest.json", checksum_manifest)
    payload["checksums"] = checksum_manifest
    _write_json(output_dir / "classification_benchmark_manifest.json", payload)
    return payload


def _latency_summary(
    dataset_dir: Path,
    manifest: Any,
    experiment_dir: Path,
    config: ClassificationConfig,
    repeats: int = 3,
) -> dict[str, Any]:
    sample = next(sample for sample in manifest.samples if sample.split == "test")
    image = np.load(dataset_dir / sample.image).astype(np.float32)
    calibration = json.loads((experiment_dir / "calibration.json").read_text(encoding="utf-8"))
    threshold_policy = json.loads(
        (experiment_dir / "threshold_policy.json").read_text(encoding="utf-8")
    )
    durations = []
    for _ in range(repeats):
        _, duration = classify_array(
            image,
            experiment_dir / "best_model.pt",
            calibration,
            threshold_policy,
            config,
        )
        durations.append(float(duration))
    return {
        "protocol": "single held-out test sample forward pass",
        "sample_id": sample.sample_id,
        "repeat_count": repeats,
        "min_seconds": min(durations),
        "median_seconds": statistics.median(durations),
        "max_seconds": max(durations),
        "mean_seconds": statistics.fmean(durations),
    }


def _comparison(model_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cnn = model_results["small_3d_cnn"]
    vit = model_results["tiny_3d_vit"]
    metric_keys = (
        "auroc",
        "auprc",
        "f1",
        "sensitivity",
        "recall",
        "specificity",
        "precision",
        "brier_score",
        "expected_calibration_error",
    )
    metric_deltas = {}
    for key in metric_keys:
        metric_deltas[key] = _delta(
            cnn["test_metrics"].get(key),
            vit["test_metrics"].get(key),
        )
    return {
        "delta_direction": "tiny_3d_vit_minus_small_3d_cnn",
        "test_metric_deltas": metric_deltas,
        "parameter_count_delta": vit["parameter_count"] - cnn["parameter_count"],
        "checkpoint_size_bytes_delta": (
            vit["checkpoint_size_bytes"] - cnn["checkpoint_size_bytes"]
        ),
        "training_duration_seconds_delta": (
            vit["training_duration_seconds"] - cnn["training_duration_seconds"]
        ),
        "latency_mean_seconds_delta": (
            vit["latency_summary"]["mean_seconds"] - cnn["latency_summary"]["mean_seconds"]
        ),
        "interpretation": (
            "Deltas are specific to this synthetic/de-identified engineering benchmark and do "
            "not establish a universally better model."
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
    return {
        "generated_at": stable_timestamp(),
        "algorithm": "sha256",
        "files": checksums,
    }


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Classification CNN Vs Vision Transformer Benchmark",
        "",
        "Synthetic/de-identified engineering benchmark only. No clinical, diagnostic, or "
        "medical-device claim is made.",
        "",
        f"- Benchmark ID: `{payload['benchmark_id']}`",
        f"- Dataset ID: `{payload['dataset_id']}`",
        f"- Random seed: `{payload['benchmark_fairness_attestation']['random_seed']}`",
        "",
        "## Fairness Controls",
    ]
    fairness = payload["benchmark_fairness_attestation"]
    lines.extend(
        [
            f"- Manifest checksum: `{fairness['classification_manifest_checksum']}`",
            f"- Split counts: `{fairness['split_counts']}`",
            f"- {fairness['shared_preprocessing_statement']}",
            f"- {fairness['shared_transforms_statement']}",
            f"- {fairness['shared_labels_statement']}",
            f"- {fairness['shared_optimizer_training_policy_statement']}",
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
                f"- Test AUROC: `{metrics.get('auroc')}`",
                f"- Test AUPRC: `{metrics.get('auprc')}`",
                f"- Test F1: `{metrics.get('f1')}`",
                f"- Test Brier score: `{metrics.get('brier_score')}`",
                f"- Latency mean seconds: `{result['latency_summary']['mean_seconds']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Limits",
            "- Results are specific to this benchmark configuration.",
            "- The comparison does not declare a universally better architecture.",
            "- Synthetic metrics do not demonstrate clinical performance.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
