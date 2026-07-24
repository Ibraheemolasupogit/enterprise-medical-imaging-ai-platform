"""Training pipeline for synthetic lesion-presence classification."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from medical_imaging_platform.classification.calibration import apply_calibration, fit_calibration
from medical_imaging_platform.classification.checkpoint import load_state_dict, save_state_dict
from medical_imaging_platform.classification.dataset import (
    ClassificationTorchDataset,
    validate_classification_dataset,
)
from medical_imaging_platform.classification.export import write_experiment_evidence
from medical_imaging_platform.classification.losses import classification_loss
from medical_imaging_platform.classification.metrics import compute_classification_metrics
from medical_imaging_platform.classification.model_factory import build_classifier, parameter_count
from medical_imaging_platform.classification.models import ClassificationConfig
from medical_imaging_platform.classification.pipeline import (
    configuration_checksum,
    dependency_versions,
    manifest_checksum,
    resolve_device,
    safe_output_dir,
    set_reproducibility,
)
from medical_imaging_platform.classification.quality import evaluate_quality
from medical_imaging_platform.classification.thresholds import select_threshold
from medical_imaging_platform.classification.transforms import build_transforms
from medical_imaging_platform.synthetic.manifest import sha256_file, stable_timestamp


def train_classification_experiment(
    dataset_dir: Path,
    *,
    output_root: Path,
    config: ClassificationConfig,
    seed: int | None = None,
    epochs: int | None = None,
    device_name: str | None = None,
    overwrite: bool,
) -> dict[str, Any]:
    """Train a small CPU-compatible binary classifier and write evidence."""
    effective = config.model_copy(
        update={
            "random_seed": config.random_seed if seed is None else seed,
            "epochs": config.epochs if epochs is None else epochs,
            "device": config.device if device_name is None else device_name,
        }
    )
    set_reproducibility(effective.random_seed)
    manifest = validate_classification_dataset(dataset_dir)
    experiment_id = f"classification-{manifest.dataset_id}-{effective.policy_version}"
    output_dir = safe_output_dir(output_root, experiment_id)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Classification experiment already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(effective.device)
    train_dataset = ClassificationTorchDataset(
        dataset_dir, manifest, "train", transform=build_transforms(effective, "train")
    )
    val_dataset = ClassificationTorchDataset(
        dataset_dir, manifest, "validation", transform=build_transforms(effective, "validation")
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective.batch_size,
        shuffle=True,
        num_workers=effective.num_workers,
        generator=torch.Generator().manual_seed(effective.random_seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_classifier(effective).to(device)
    positive_weight = _positive_weight(train_dataset, effective)
    optimiser_class = torch.optim.AdamW if effective.optimizer == "adamw" else torch.optim.Adam
    optimiser = optimiser_class(
        model.parameters(), lr=effective.learning_rate, weight_decay=effective.weight_decay
    )
    history: list[dict[str, float | int]] = []
    best_score = -1.0
    best_epoch = 0
    patience = 0
    start = time.perf_counter()
    for epoch in range(1, effective.epochs + 1):
        train_loss = _train_epoch(
            model, train_loader, optimiser, effective, positive_weight, device
        )
        validation = _evaluate_loader(model, val_loader, effective, positive_weight, device)
        validation_auprc = float(validation["metrics"]["auprc"] or 0.0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": float(validation["loss"]),
                "validation_auroc": float(validation["metrics"]["auroc"] or 0.0),
                "validation_auprc": validation_auprc,
                "validation_recall": float(validation["metrics"]["recall"] or 0.0),
            }
        )
        save_state_dict(output_dir / "last_model.pt", model)
        if validation_auprc >= best_score:
            best_score = validation_auprc
            best_epoch = epoch
            patience = 0
            save_state_dict(output_dir / "best_model.pt", model)
        else:
            patience += 1
        if effective.early_stopping_patience and patience >= effective.early_stopping_patience:
            break
    duration = float(time.perf_counter() - start)
    best_model = build_classifier(effective).to(device)
    best_model.load_state_dict(load_state_dict(output_dir / "best_model.pt"))
    validation_eval = evaluate_split(
        dataset_dir, manifest, "validation", best_model, effective, positive_weight, device
    )
    calibration = fit_calibration(validation_eval["logits"], validation_eval["labels"], effective)
    calibrated_validation = [
        apply_calibration(logit, calibration) for logit in validation_eval["logits"]
    ]
    threshold_policy = select_threshold(validation_eval["labels"], calibrated_validation, effective)
    validation_eval = _with_calibrated_metrics(
        validation_eval, calibrated_validation, threshold_policy
    )
    test_eval = evaluate_split(
        dataset_dir, manifest, "test", best_model, effective, positive_weight, device
    )
    calibrated_test = [apply_calibration(logit, calibration) for logit in test_eval["logits"]]
    test_eval = _with_calibrated_metrics(test_eval, calibrated_test, threshold_policy)
    predictions = validation_eval["predictions"] + test_eval["predictions"]
    evaluation = {
        "validation": validation_eval,
        "test": test_eval,
        "calibration_present": True,
        "threshold_present": True,
    }
    checkpoint_paths = [output_dir / "best_model.pt", output_dir / "last_model.pt"]
    status, findings = evaluate_quality(
        config=effective,
        training_history=history,
        evaluation=evaluation,
        checkpoint_paths=checkpoint_paths,
        evidence_paths=[],
    )
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "dataset_id": manifest.dataset_id,
        "dataset_manifest_checksum": manifest_checksum(
            dataset_dir / "classification_manifest.json"
        ),
        "configuration_checksum": configuration_checksum(effective),
        "experiment_config": effective.model_dump(mode="json"),
        "training_history": history,
        "evaluation": evaluation,
        "predictions": predictions,
        "calibration": calibration,
        "threshold_policy": threshold_policy,
        "status": status,
        "quality_findings": [finding.model_dump(mode="json") for finding in findings],
        "device": str(device),
        "random_seed": effective.random_seed,
        "dependency_versions": dependency_versions(),
        "model_architecture": {
            "family": "small_3d_cnn",
            "channels": list(effective.channels),
            "dropout": effective.dropout,
            "output_logits": 1,
        },
        "model_parameter_count": parameter_count(best_model),
        "class_counts": manifest.class_counts,
        "class_weight": positive_weight,
        "best_epoch": best_epoch,
        "best_validation_auprc": best_score,
        "training_duration_seconds": duration,
        "generated_at": stable_timestamp(),
        "paths": {
            "best_model": "best_model.pt",
            "last_model": "last_model.pt",
            "calibration": "calibration.json",
            "threshold_policy": "threshold_policy.json",
            "report": "classification_report.md",
        },
        "checksums": {
            "best_model": sha256_file(output_dir / "best_model.pt"),
            "last_model": sha256_file(output_dir / "last_model.pt"),
        },
        "limitations": [
            "Synthetic lesion-presence labels only.",
            "Not benign-versus-malignant classification.",
            "No clinical-performance claim.",
        ],
        "recommended_next_action": _recommended_next_action(status),
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
    config: ClassificationConfig,
    positive_weight: float,
    device: torch.device,
) -> dict[str, Any]:
    dataset = ClassificationTorchDataset(
        dataset_dir, manifest, split, transform=build_transforms(config, split)
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return _evaluate_loader(model, loader, config, positive_weight, device, samples=dataset.samples)


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimiser: torch.optim.Optimizer,
    config: ClassificationConfig,
    positive_weight: float,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        optimiser.zero_grad(set_to_none=True)
        loss = classification_loss(model(image), label, config, positive_weight)
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    config: ClassificationConfig,
    positive_weight: float,
    device: torch.device,
    samples: list[Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    labels: list[int] = []
    logits: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            output = model(image)
            losses.append(float(classification_loss(output, label, config, positive_weight).cpu()))
            labels.extend(int(value) for value in label.cpu().numpy().reshape(-1))
            logits.extend(float(value) for value in output.cpu().numpy().reshape(-1))
    probabilities = (1.0 / (1.0 + np.exp(-np.asarray(logits)))).tolist()
    predictions = []
    if samples is not None:
        for sample, probability, label, logit in zip(
            samples, probabilities, labels, logits, strict=True
        ):
            predictions.append(
                {
                    "sample_id": sample.sample_id,
                    "case_id": sample.case_id,
                    "split": sample.split,
                    "label": label,
                    "raw_logit": logit,
                    "raw_probability": probability,
                }
            )
    return {
        "loss": float(np.mean(losses)),
        "labels": labels,
        "logits": logits,
        "probabilities": probabilities,
        "metrics": compute_classification_metrics(labels, probabilities, 0.5),
        "predictions": predictions,
    }


def _with_calibrated_metrics(
    evaluation: dict[str, Any],
    calibrated: list[float],
    threshold_policy: dict[str, object],
) -> dict[str, Any]:
    threshold = _threshold_value(threshold_policy)
    metrics = compute_classification_metrics(evaluation["labels"], calibrated, threshold)
    updated_predictions = []
    for prediction, probability in zip(evaluation["predictions"], calibrated, strict=True):
        item = dict(prediction)
        item["calibrated_probability"] = probability
        item["threshold"] = threshold
        item["predicted_label"] = int(probability >= threshold)
        updated_predictions.append(item)
    return {
        **evaluation,
        "calibrated_probabilities": calibrated,
        "metrics": metrics,
        "predictions": updated_predictions,
    }


def _positive_weight(dataset: ClassificationTorchDataset, config: ClassificationConfig) -> float:
    if config.positive_class_weight is not None:
        return config.positive_class_weight
    positives = sum(sample.label for sample in dataset.samples)
    negatives = len(dataset.samples) - positives
    if positives == 0:
        return 1.0
    return float(negatives / positives)


def _recommended_next_action(status: str) -> str:
    if status == "PASS":
        return "Proceed only to research engineering review and Milestone 10 planning."
    if status == "PASS_WITH_WARNINGS":
        return "Review classification warnings before downstream research use."
    return "Do not use this classifier downstream without remediation."


def _threshold_value(threshold_policy: dict[str, object]) -> float:
    value = threshold_policy.get("selected_threshold")
    if isinstance(value, int | float | str):
        return float(value)
    raise ValueError("Threshold policy is missing a numeric selected_threshold.")
