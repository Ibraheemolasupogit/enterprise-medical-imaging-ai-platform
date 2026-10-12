"""Training and evaluation for synthetic 2D lesion detection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from medical_imaging_platform.detection.checkpoint import load_state_dict, save_state_dict
from medical_imaging_platform.detection.dataset import (
    DetectionTorchDataset,
    collate_detection_batch,
    validate_detection_dataset,
)
from medical_imaging_platform.detection.export import write_experiment_evidence
from medical_imaging_platform.detection.metrics import compute_detection_metrics
from medical_imaging_platform.detection.model_factory import (
    architecture_metadata,
    build_detector,
    parameter_count,
    predictions_to_common,
)
from medical_imaging_platform.detection.models import DetectionConfig
from medical_imaging_platform.detection.pipeline import (
    configuration_checksum,
    dependency_versions,
    manifest_checksum,
    resolve_device,
    safe_output_dir,
    set_reproducibility,
)
from medical_imaging_platform.synthetic.manifest import sha256_file, stable_timestamp


def train_detection_experiment(
    dataset_dir: Path,
    *,
    output_root: Path,
    config: DetectionConfig,
    seed: int | None = None,
    epochs: int | None = None,
    device_name: str | None = None,
    overwrite: bool,
) -> dict[str, Any]:
    """Train one detection model family and write deterministic evidence."""
    effective = config.model_copy(
        update={
            "random_seed": config.random_seed if seed is None else seed,
            "epochs": config.epochs if epochs is None else epochs,
            "device": config.device if device_name is None else device_name,
        }
    )
    set_reproducibility(effective.random_seed)
    manifest = validate_detection_dataset(dataset_dir)
    experiment_id = _experiment_id(manifest.dataset_id, effective)
    output_dir = safe_output_dir(output_root, experiment_id)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Detection experiment already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(effective.device)

    train_dataset = DetectionTorchDataset(dataset_dir, manifest, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective.batch_size,
        shuffle=True,
        num_workers=effective.num_workers,
        collate_fn=collate_detection_batch,
        generator=torch.Generator().manual_seed(effective.random_seed),
    )
    model = build_detector(effective).to(device)
    optimiser_class = torch.optim.AdamW if effective.optimizer == "adamw" else torch.optim.Adam
    optimiser = optimiser_class(
        model.parameters(), lr=effective.learning_rate, weight_decay=effective.weight_decay
    )
    history: list[dict[str, float | int]] = []
    best_map = -1.0
    best_epoch = 0
    start = time.perf_counter()
    for epoch in range(1, effective.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimiser, device)
        validation_eval = evaluate_split(
            dataset_dir, manifest, "validation", model, effective, device
        )
        validation_map = float(validation_eval["metrics"]["map"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_map": validation_map,
                "validation_ap50": float(validation_eval["metrics"]["ap50"]),
            }
        )
        save_state_dict(output_dir / "last_model.pt", model)
        if validation_map >= best_map:
            best_map = validation_map
            best_epoch = epoch
            save_state_dict(output_dir / "best_model.pt", model)
    duration = float(time.perf_counter() - start)
    best_model = build_detector(effective).to(device)
    best_model.load_state_dict(load_state_dict(output_dir / "best_model.pt"), strict=False)
    validation_eval = evaluate_split(
        dataset_dir, manifest, "validation", best_model, effective, device
    )
    test_eval = evaluate_split(dataset_dir, manifest, "test", best_model, effective, device)
    status, findings = _quality(effective, validation_eval, test_eval)
    predictions = validation_eval["predictions"] + test_eval["predictions"]
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "dataset_id": manifest.dataset_id,
        "dataset_manifest_checksum": manifest_checksum(
            dataset_dir / "detection_dataset_manifest.json"
        ),
        "configuration_checksum": configuration_checksum(effective),
        "experiment_config": effective.model_dump(mode="json"),
        "training_history": history,
        "evaluation": {"validation": validation_eval, "test": test_eval},
        "predictions": predictions,
        "status": status,
        "quality_findings": findings,
        "device": str(device),
        "random_seed": effective.random_seed,
        "dependency_versions": dependency_versions(),
        "model_family": effective.model_family,
        "model_architecture": architecture_metadata(effective, best_model),
        "model_parameter_count": parameter_count(best_model),
        "split_counts": manifest.split_counts,
        "positive_image_count": manifest.positive_image_count,
        "negative_image_count": manifest.negative_image_count,
        "box_count": manifest.box_count,
        "best_epoch": best_epoch,
        "best_validation_map": best_map,
        "training_duration_seconds": duration,
        "generated_at": stable_timestamp(),
        "paths": {
            "best_model": "best_model.pt",
            "last_model": "last_model.pt",
            "experiment_config": "experiment_config.json",
            "training_history": "training_history.json",
            "evaluation": "detection_evaluation.json",
            "predictions": "detection_predictions.json",
            "model_manifest": "detection_model_manifest.json",
            "report": "detection_report.md",
            "checksum_manifest": "evidence/checksum_manifest.json",
        },
        "checksums": {
            "best_model": sha256_file(output_dir / "best_model.pt"),
            "last_model": sha256_file(output_dir / "last_model.pt"),
        },
        "limitations": [
            "Synthetic 2D axial lesion boxes only.",
            "2D boxes do not represent full 3D lesion extent.",
            "Detector confidence is not a clinical probability.",
            "No diagnosis, triage, RECIST, patient-management, or medical-device claim.",
        ],
    }
    evidence_checksums = write_experiment_evidence(output_dir, payload)
    payload["checksums"] = {**payload["checksums"], **evidence_checksums}
    write_experiment_evidence(output_dir, payload)
    return payload


def evaluate_split(
    dataset_dir: Path,
    manifest: Any,
    split: str,
    model: torch.nn.Module,
    config: DetectionConfig,
    device: torch.device,
) -> dict[str, Any]:
    dataset = DetectionTorchDataset(dataset_dir, manifest, split)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_detection_batch
    )
    return _evaluate_loader(model, loader, config, device)


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        images = [image.to(device) for image in batch["images"]]
        targets = [_target_to_device(target, device) for target in batch["targets"]]
        optimiser.zero_grad(set_to_none=True)
        loss_dict = (
            model.loss(images, targets) if hasattr(model, "loss") else model(images, targets)
        )
        if not isinstance(loss_dict, dict):
            raise RuntimeError("Detection training must return a loss dictionary.")
        loss = sum(value for value in loss_dict.values())
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    config: DetectionConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    ground_truths: list[dict[str, Any]] = []
    latencies: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = [image.to(device) for image in batch["images"]]
            start = time.perf_counter()
            if hasattr(model, "predict"):
                raw_predictions = model.predict(images)
            else:
                raw_predictions = model(images)
            latencies.append(float(time.perf_counter() - start))
            common_predictions = predictions_to_common(
                cast(list[dict[str, torch.Tensor]], raw_predictions)
            )
            for prediction, target, sample in zip(
                common_predictions, batch["targets"], batch["samples"], strict=True
            ):
                predictions.append({"sample_id": sample.sample_id, **prediction})
                ground_truths.append(
                    {
                        "sample_id": sample.sample_id,
                        "boxes": target["boxes"].cpu().numpy().astype(float).tolist(),
                        "labels": target["labels"].cpu().numpy().astype(int).tolist(),
                    }
                )
    metrics = compute_detection_metrics(
        predictions, ground_truths, iou_thresholds=config.evaluation_iou_thresholds
    )
    metrics["latency_summary"] = _latency_summary(latencies)
    return {"metrics": metrics, "predictions": predictions, "ground_truths": ground_truths}


def _target_to_device(
    target: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in target.items()}


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min_seconds": 0.0, "mean_seconds": 0.0, "max_seconds": 0.0}
    return {
        "count": len(values),
        "min_seconds": float(min(values)),
        "mean_seconds": float(np.mean(values)),
        "max_seconds": float(max(values)),
    }


def _quality(
    config: DetectionConfig, validation_eval: dict[str, Any], test_eval: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    findings = []
    validation_map = float(validation_eval["metrics"]["map"])
    test_fp = float(test_eval["metrics"]["false_positives_per_image"])
    findings.append(
        {
            "rule_id": "DET-QC-MAP-001",
            "severity": "INFO" if validation_map >= config.minimum_validation_map else "ERROR",
            "status": "PASS" if validation_map >= config.minimum_validation_map else "FAIL",
            "message": "Validation mAP meets configured engineering threshold.",
            "observed_value": validation_map,
            "expected_value": config.minimum_validation_map,
            "remediation": "Review dataset, threshold, or training configuration.",
        }
    )
    findings.append(
        {
            "rule_id": "DET-QC-FP-001",
            "severity": "INFO" if test_fp <= config.maximum_false_positives_per_image else "ERROR",
            "status": "PASS" if test_fp <= config.maximum_false_positives_per_image else "FAIL",
            "message": "False positives per image is within configured engineering threshold.",
            "observed_value": test_fp,
            "expected_value": config.maximum_false_positives_per_image,
            "remediation": "Review score threshold or model configuration.",
        }
    )
    status = "PASS" if all(finding["status"] == "PASS" for finding in findings) else "FAIL"
    return status, findings


def _experiment_id(dataset_id: str, config: DetectionConfig) -> str:
    return (
        f"detection-{config.model_family}-{dataset_id}-{config.policy_version}-"
        f"seed-{config.random_seed}"
    )
