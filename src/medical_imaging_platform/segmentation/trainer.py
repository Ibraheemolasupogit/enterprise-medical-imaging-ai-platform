"""CPU-compatible MONAI segmentation training pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from medical_imaging_platform.segmentation.checkpoint import save_state_dict
from medical_imaging_platform.segmentation.dataset import (
    SegmentationTorchDataset,
    validate_segmentation_dataset,
)
from medical_imaging_platform.segmentation.export import write_experiment_evidence
from medical_imaging_platform.segmentation.inference import evaluate_model_on_split
from medical_imaging_platform.segmentation.losses import segmentation_loss
from medical_imaging_platform.segmentation.model_factory import build_unet, parameter_count
from medical_imaging_platform.segmentation.models import SegmentationConfig
from medical_imaging_platform.segmentation.pipeline import (
    configuration_checksum,
    dependency_versions,
    manifest_checksum,
    resolve_device,
    safe_experiment_dir,
    set_reproducibility,
)
from medical_imaging_platform.segmentation.quality import evaluate_quality
from medical_imaging_platform.segmentation.transforms import build_transforms
from medical_imaging_platform.synthetic.manifest import sha256_file, stable_timestamp


def train_segmentation_experiment(
    dataset_dir: Path,
    *,
    output_root: Path,
    config: SegmentationConfig,
    seed: int | None = None,
    epochs: int | None = None,
    device_name: str | None = None,
    overwrite: bool,
) -> dict[str, Any]:
    """Train and evaluate the small 3D U-Net segmentation baseline."""
    effective = config.model_copy(
        update={
            "random_seed": config.random_seed if seed is None else seed,
            "epochs": config.epochs if epochs is None else epochs,
            "device": config.device if device_name is None else device_name,
        }
    )
    set_reproducibility(effective.random_seed)
    manifest = validate_segmentation_dataset(dataset_dir)
    experiment_id = f"segmentation-{manifest.dataset_id}-{effective.policy_version}"
    output_dir = safe_experiment_dir(output_root, experiment_id)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Segmentation experiment already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(effective.device)
    train_dataset = SegmentationTorchDataset(
        dataset_dir, manifest, "train", transform=build_transforms(effective, "train")
    )
    val_dataset = SegmentationTorchDataset(
        dataset_dir, manifest, "validation", transform=build_transforms(effective, "validation")
    )
    generator = torch.Generator().manual_seed(effective.random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective.batch_size,
        shuffle=True,
        num_workers=effective.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=effective.num_workers,
    )
    model = build_unet(effective).to(device)
    optimiser_class = torch.optim.AdamW if effective.optimizer == "adamw" else torch.optim.Adam
    optimiser = optimiser_class(
        model.parameters(), lr=effective.learning_rate, weight_decay=effective.weight_decay
    )
    history: list[dict[str, float | int]] = []
    best_dice = -1.0
    best_epoch = 0
    patience_count = 0
    start = time.perf_counter()
    for epoch in range(1, effective.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimiser, effective, device)
        validation_loss, validation_dice = _validate_epoch(model, val_loader, effective, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_dice": validation_dice,
            }
        )
        save_state_dict(output_dir / "last_model.pt", model)
        if validation_dice >= best_dice:
            best_dice = validation_dice
            best_epoch = epoch
            patience_count = 0
            save_state_dict(output_dir / "best_model.pt", model)
        else:
            patience_count += 1
        if (
            effective.early_stopping_patience > 0
            and patience_count >= effective.early_stopping_patience
        ):
            break
    duration = float(time.perf_counter() - start)
    best_checkpoint = output_dir / "best_model.pt"
    evaluation = {
        "validation": evaluate_model_on_split(
            dataset_dir, manifest, "validation", best_checkpoint, effective
        ),
        "test": evaluate_model_on_split(dataset_dir, manifest, "test", best_checkpoint, effective),
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
        "dataset_manifest_checksum": manifest_checksum(dataset_dir / "segmentation_manifest.json"),
        "configuration_checksum": configuration_checksum(effective),
        "experiment_config": effective.model_dump(mode="json"),
        "training_history": history,
        "evaluation": evaluation,
        "status": status,
        "quality_findings": [finding.model_dump(mode="json") for finding in findings],
        "device": str(device),
        "random_seed": effective.random_seed,
        "dependency_versions": dependency_versions(),
        "model_architecture": {
            "family": "MONAI UNet",
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": list(effective.channels),
            "strides": list(effective.strides),
            "num_res_units": effective.num_res_units,
        },
        "model_parameter_count": parameter_count(model),
        "best_epoch": best_epoch,
        "best_validation_dice": best_dice,
        "training_duration_seconds": duration,
        "generated_at": stable_timestamp(),
        "paths": {
            "best_model": "best_model.pt",
            "last_model": "last_model.pt",
            "report": "segmentation_report.md",
        },
        "checksums": {
            "best_model": sha256_file(output_dir / "best_model.pt"),
            "last_model": sha256_file(output_dir / "last_model.pt"),
        },
        "limitations": [
            "Synthetic engineering lesion masks only.",
            "No clinical lesion segmentation performance claim.",
            "Exact floating-point equality may vary across hardware and PyTorch builds.",
        ],
        "recommended_next_action": _recommended_next_action(status),
    }
    evidence_checksums = write_experiment_evidence(output_dir, payload)
    payload["checksums"] = {**payload["checksums"], **evidence_checksums}
    write_experiment_evidence(output_dir, payload)
    return payload


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimiser: torch.optim.Optimizer,
    config: SegmentationConfig,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        optimiser.zero_grad(set_to_none=True)
        loss = segmentation_loss(model(image), mask, config)
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    config: SegmentationConfig,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    dices: list[float] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            logits = model(image)
            losses.append(float(segmentation_loss(logits, mask, config).cpu()))
            pred = (torch.sigmoid(logits) >= config.threshold).float()
            intersection = torch.sum(pred * mask)
            denominator = torch.sum(pred) + torch.sum(mask)
            dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
            dices.append(float(dice.cpu()))
    return float(np.mean(losses)), float(np.mean(dices))


def _recommended_next_action(status: str) -> str:
    if status == "PASS":
        return "Proceed only to research engineering review and Milestone 9 planning."
    if status == "PASS_WITH_WARNINGS":
        return "Review model-quality warnings before downstream research use."
    return "Do not use this segmentation checkpoint downstream without remediation."
