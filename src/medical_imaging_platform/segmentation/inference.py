"""Segmentation inference and evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from medical_imaging_platform.segmentation.checkpoint import load_state_dict
from medical_imaging_platform.segmentation.dataset import SegmentationTorchDataset
from medical_imaging_platform.segmentation.metrics import (
    aggregate_metrics,
    compute_segmentation_metrics,
)
from medical_imaging_platform.segmentation.model_factory import build_unet
from medical_imaging_platform.segmentation.models import (
    SegmentationConfig,
    SegmentationDatasetManifest,
)
from medical_imaging_platform.segmentation.pipeline import resolve_device, set_reproducibility
from medical_imaging_platform.segmentation.postprocessing import postprocess_probability_map
from medical_imaging_platform.segmentation.transforms import build_transforms


def segment_volume(
    input_volume: Path,
    *,
    checkpoint_path: Path,
    output_dir: Path,
    config: SegmentationConfig,
    threshold: float | None = None,
    overwrite: bool,
) -> dict[str, Any]:
    """Run CPU inference for one prepared ROI volume."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Inference output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image = np.load(input_volume).astype(np.float32)
    if image.shape != config.input_shape or not np.all(np.isfinite(image)):
        raise ValueError("Input volume must match configured shape and contain finite values.")
    probability, duration = predict_probability(image, checkpoint_path, config)
    mask, warnings, counts = postprocess_probability_map(
        probability, config=config, threshold=threshold
    )
    _write_numpy(output_dir / "probability_map.npy", probability.astype(np.float32))
    _write_numpy(output_dir / "predicted_mask.npy", mask.astype(np.uint8))
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "input_volume": str(input_volume),
        "threshold": config.threshold if threshold is None else threshold,
        "inference_duration_seconds": duration,
        "postprocessing": counts,
        "warnings": warnings,
    }
    (output_dir / "inference_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def predict_probability(
    image: np.ndarray, checkpoint_path: Path, config: SegmentationConfig
) -> tuple[np.ndarray, float]:
    """Predict one probability map."""
    set_reproducibility(config.random_seed)
    device = resolve_device(config.device)
    model = build_unet(config).to(device)
    model.load_state_dict(load_state_dict(checkpoint_path))
    model.eval()
    tensor = torch.as_tensor(image[None, None], dtype=torch.float32, device=device)
    start = time.perf_counter()
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor)).cpu().numpy()[0, 0]
    return probability.astype(np.float32), float(time.perf_counter() - start)


def evaluate_model_on_split(
    dataset_dir: Path,
    manifest: SegmentationDatasetManifest,
    split: str,
    checkpoint_path: Path,
    config: SegmentationConfig,
) -> dict[str, Any]:
    """Evaluate a checkpoint on one prepared split."""
    dataset = SegmentationTorchDataset(
        dataset_dir, manifest, split, transform=build_transforms(config, split)
    )
    per_case: list[dict[str, Any]] = []
    metric_sets = []
    for index, sample in enumerate(dataset.samples):
        item = dataset[index]
        image = item["image"].numpy()[0]
        target = item["mask"].numpy()[0].astype(bool)
        probability, _ = predict_probability(image, checkpoint_path, config)
        prediction, warnings, _ = postprocess_probability_map(probability, config=config)
        metrics = compute_segmentation_metrics(
            prediction,
            target,
            spacing_mm=(
                float(sample.spacing_mm[0]),
                float(sample.spacing_mm[1]),
                float(sample.spacing_mm[2]),
            ),
        )
        metric_sets.append(metrics)
        per_case.append(
            {
                "sample_id": sample.sample_id,
                "case_id": sample.case_id,
                "scenario": sample.scenario,
                "positive_case": sample.lesion_volume_voxels > 0,
                "metrics": metrics.model_dump(mode="json"),
                "warnings": warnings,
            }
        )
    positive_metrics = [
        item
        for item, sample in zip(metric_sets, dataset.samples, strict=True)
        if sample.lesion_volume_voxels > 0
    ]
    negative_metrics = [
        item
        for item, sample in zip(metric_sets, dataset.samples, strict=True)
        if sample.lesion_volume_voxels == 0
    ]
    return {
        "split": split,
        "case_count": len(per_case),
        "per_case": per_case,
        "aggregate": aggregate_metrics(metric_sets),
        "positive_case_aggregate": aggregate_metrics(positive_metrics),
        "negative_case_aggregate": aggregate_metrics(negative_metrics),
        "scenario_metrics": _scenario_metrics(per_case),
        "failure_cases": [
            item["sample_id"]
            for item in per_case
            if item["metrics"]["recall"] is not None and item["metrics"]["recall"] < 0.5
        ],
        "false_positive_voxels_max": max(
            (item.false_positive_voxels for item in metric_sets), default=0
        ),
        "relative_volume_error_max": max(
            (
                item.relative_volume_error
                for item in metric_sets
                if item.relative_volume_error is not None
            ),
            default=0.0,
        ),
    }


def _scenario_metrics(per_case: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for scenario in sorted({item["scenario"] for item in per_case}):
        metrics = [
            item["metrics"]["dice"]
            for item in per_case
            if item["scenario"] == scenario and item["metrics"]["dice"] is not None
        ]
        output[scenario] = {
            "case_count": sum(1 for item in per_case if item["scenario"] == scenario),
            "mean_dice": float(np.mean(metrics)) if metrics else None,
        }
    return output


def _write_numpy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.replace(path)
