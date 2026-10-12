"""Synthetic 2D axial detection dataset preparation and loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from medical_imaging_platform.detection.models import (
    DetectionConfig,
    DetectionDatasetManifest,
    DetectionSample,
)
from medical_imaging_platform.synthetic.manifest import (
    MANIFEST_FILENAME,
    load_manifest,
    sha256_file,
    stable_timestamp,
    write_json,
)
from medical_imaging_platform.synthetic.split import SPLIT_FILENAME, DatasetSplits
from medical_imaging_platform.synthetic.validation import detect_subject_leakage


class DetectionDataError(ValueError):
    """Raised when detection data are invalid."""


def mask_to_boxes_xyxy(mask_2d: np.ndarray, minimum_pixels: int = 1) -> np.ndarray:
    """Return one half-open xyxy lesion box for a binary 2D mask, or no boxes."""
    array = np.asarray(mask_2d)
    if array.ndim != 2:
        raise DetectionDataError("Detection mask slice must be 2D.")
    if not set(np.unique(array)).issubset({0, 1}):
        raise DetectionDataError("Detection mask slice must be binary.")
    binary = array.astype(bool)
    if int(np.count_nonzero(binary)) < minimum_pixels:
        return np.zeros((0, 4), dtype=np.float32)
    yy, xx = np.where(binary)
    return np.asarray(
        [[float(xx.min()), float(yy.min()), float(xx.max() + 1), float(yy.max() + 1)]],
        dtype=np.float32,
    )


def prepare_detection_dataset(
    synthetic_dataset_dir: Path,
    *,
    output_root: Path,
    config: DetectionConfig,
    overwrite: bool,
) -> DetectionDatasetManifest:
    """Create deterministic 2D axial detection samples from synthetic volumes and lesion masks."""
    source_root = synthetic_dataset_dir.resolve()
    manifest = load_manifest(source_root / MANIFEST_FILENAME)
    splits = DatasetSplits.model_validate_json(
        (source_root / SPLIT_FILENAME).read_text(encoding="utf-8")
    )
    detect_subject_leakage(
        splits.splits,
        {record.case_id: record.research_subject_id for record in manifest.records},
    )
    case_to_split = {
        case_id: split_name
        for split_name, case_ids in splits.splits.items()
        for case_id in case_ids
    }
    dataset_id = f"detection-{manifest.dataset_id}-{config.policy_version}"
    root = _safe_child(output_root, dataset_id)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise DetectionDataError(f"Detection dataset already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)

    samples: list[DetectionSample] = []
    for record in sorted(manifest.records, key=lambda item: item.case_id):
        split = case_to_split.get(record.case_id)
        if split not in {"train", "validation", "test"}:
            raise DetectionDataError(f"Case missing from split file: {record.case_id}")
        for timepoint in ("previous", "current"):
            volume_key = f"{timepoint}_volume"
            mask_key = f"{timepoint}_lesion_mask"
            volume = _load_array(source_root, record.file_paths[volume_key])
            mask = _load_array(source_root, record.file_paths[mask_key])
            _validate_source(volume, mask, config.image_shape)
            positive_z = _positive_slices(mask, config)
            negative_z = _negative_slices(mask, config, exclude=set(positive_z))
            for z_index in [*positive_z, *negative_z]:
                image = _normalise_slice(volume[int(z_index)])
                boxes = mask_to_boxes_xyxy(mask[int(z_index)], config.minimum_lesion_pixels)
                labels = [1] * int(boxes.shape[0])
                sample_id = f"{record.case_id}-{timepoint}-z{int(z_index):03d}"
                sample_dir = root / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)
                _write_numpy(sample_dir / "image.npy", image.astype(np.float32))
                samples.append(
                    DetectionSample(
                        sample_id=sample_id,
                        image=str(Path(sample_id) / "image.npy"),
                        boxes=boxes.astype(float).tolist(),
                        labels=labels,
                        case_id=record.case_id,
                        research_subject_id=record.research_subject_id,
                        split=cast(Any, split),
                        scenario=record.scenario,
                        timepoint=cast(Any, timepoint),
                        source_z=int(z_index),
                        source_volume_shape=cast(
                            tuple[int, int, int], tuple(int(axis) for axis in volume.shape)
                        ),
                        image_shape=config.image_shape,
                        spacing_mm=record.voxel_spacing_mm,
                        source_checksums={
                            volume_key: record.checksums[volume_key],
                            mask_key: record.checksums[mask_key],
                        },
                        positive=bool(labels),
                    )
                )
    samples = sorted(samples, key=lambda item: item.sample_id)
    prepared = DetectionDatasetManifest(
        dataset_id=dataset_id,
        source_dataset_id=manifest.dataset_id,
        policy_version=config.policy_version,
        generated_at=stable_timestamp(),
        image_shape=config.image_shape,
        coordinate_semantics=(
            "2D half-open xyxy pixel boxes on axial slices: "
            "0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height. "
            "source_z links each 2D sample back to the canonical z,y,x volume."
        ),
        samples=samples,
        split_counts={
            split_name: sum(1 for sample in samples if sample.split == split_name)
            for split_name in ("train", "validation", "test")
        },
        positive_image_count=sum(1 for sample in samples if sample.positive),
        negative_image_count=sum(1 for sample in samples if not sample.positive),
        box_count=sum(len(sample.boxes) for sample in samples),
        checksums={sample.sample_id: sha256_file(root / sample.image) for sample in samples},
        label_semantics={"1": "synthetic_lesion_box", "background": "no visible synthetic lesion"},
    )
    write_json(root / "detection_dataset_manifest.json", prepared.model_dump(mode="json"))
    return validate_detection_dataset(root)


def validate_detection_dataset(dataset_dir: Path) -> DetectionDatasetManifest:
    """Validate prepared detection samples, checksums, boxes, and split isolation."""
    manifest_path = dataset_dir / "detection_dataset_manifest.json"
    if not manifest_path.exists():
        raise DetectionDataError("Detection dataset manifest is missing.")
    manifest = DetectionDatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    ordered = [sample.sample_id for sample in manifest.samples]
    if ordered != sorted(ordered):
        raise DetectionDataError("Detection samples must be deterministically ordered.")
    split_subjects: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    positives = 0
    negatives = 0
    boxes = 0
    for sample in manifest.samples:
        image = np.load(dataset_dir / sample.image)
        _validate_image(image, manifest.image_shape)
        if sha256_file(dataset_dir / sample.image) != manifest.checksums[sample.sample_id]:
            raise DetectionDataError(f"Checksum mismatch for sample {sample.sample_id}")
        DetectionSample.model_validate(sample.model_dump(mode="json"))
        split_subjects[sample.split].add(sample.research_subject_id)
        positives += int(sample.positive)
        negatives += int(not sample.positive)
        boxes += len(sample.boxes)
    for first_name, first_subjects in split_subjects.items():
        for second_name, second_subjects in split_subjects.items():
            if first_name < second_name and first_subjects & second_subjects:
                raise DetectionDataError("Research subject leakage detected.")
    if positives != manifest.positive_image_count or negatives != manifest.negative_image_count:
        raise DetectionDataError("Positive/negative image counts do not match manifest.")
    if boxes != manifest.box_count:
        raise DetectionDataError("Box count does not match manifest.")
    return manifest


class DetectionTorchDataset:
    """Torch-compatible 2D detection dataset."""

    def __init__(self, dataset_dir: Path, manifest: DetectionDatasetManifest, split: str) -> None:
        self.dataset_dir = dataset_dir
        self.samples = [sample for sample in manifest.samples if sample.split == split]
        if not self.samples:
            raise DetectionDataError(f"No detection samples for split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = np.load(self.dataset_dir / sample.image).astype(np.float32)
        boxes = torch.as_tensor(sample.boxes, dtype=torch.float32)
        labels = torch.as_tensor(sample.labels, dtype=torch.int64)
        return {
            "image": torch.as_tensor(image[None], dtype=torch.float32),
            "target": {
                "boxes": boxes.reshape((-1, 4)),
                "labels": labels.reshape((-1,)),
                "image_id": torch.as_tensor([index], dtype=torch.int64),
            },
            "sample": sample,
        }


def collate_detection_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "targets": [item["target"] for item in batch],
        "samples": [item["sample"] for item in batch],
    }


def _positive_slices(mask: np.ndarray, config: DetectionConfig) -> list[int]:
    counts = np.count_nonzero(mask.astype(bool), axis=(1, 2))
    indices = [
        int(index) for index, count in enumerate(counts) if count >= config.minimum_lesion_pixels
    ]
    return _bounded_even(indices, config.max_positive_slices_per_volume)


def _negative_slices(mask: np.ndarray, config: DetectionConfig, exclude: set[int]) -> list[int]:
    if config.max_negative_slices_per_volume == 0:
        return []
    indices = [
        int(index)
        for index, count in enumerate(np.count_nonzero(mask.astype(bool), axis=(1, 2)))
        if count == 0 and int(index) not in exclude
    ]
    return _bounded_even(indices, config.max_negative_slices_per_volume)


def _bounded_even(indices: list[int], limit: int) -> list[int]:
    if len(indices) <= limit:
        return indices
    positions = np.linspace(0, len(indices) - 1, limit).round().astype(int).tolist()
    return [indices[position] for position in sorted(set(positions))]


def _normalise_slice(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise DetectionDataError("Image contains no finite pixels.")
    lower = float(np.percentile(finite, 1))
    upper = float(np.percentile(finite, 99))
    if upper <= lower:
        return np.zeros_like(image, dtype=np.float32)
    normalised = np.clip((image.astype(np.float32) - lower) / (upper - lower), 0.0, 1.0).astype(
        np.float32
    )
    return cast(np.ndarray, normalised)


def _validate_source(volume: np.ndarray, mask: np.ndarray, image_shape: tuple[int, int]) -> None:
    if volume.ndim != 3 or mask.ndim != 3 or volume.shape != mask.shape:
        raise DetectionDataError("Source volume and lesion mask must be matching 3D arrays.")
    if tuple(volume.shape[1:]) != image_shape:
        raise DetectionDataError("Source axial shape does not match detection image_shape.")
    if not np.all(np.isfinite(volume)):
        raise DetectionDataError("Source volume contains non-finite values.")
    if not set(np.unique(mask)).issubset({0, 1}):
        raise DetectionDataError("Source lesion mask must be binary.")


def _validate_image(image: np.ndarray, image_shape: tuple[int, int]) -> None:
    if image.shape != image_shape or image.ndim != 2:
        raise DetectionDataError("Detection image must match configured 2D image_shape.")
    if image.dtype != np.float32:
        raise DetectionDataError("Detection image must be float32.")
    if not np.all(np.isfinite(image)):
        raise DetectionDataError("Detection image contains non-finite values.")


def _load_array(root: Path, relative_path: str) -> np.ndarray:
    path = (root / relative_path).resolve()
    if os.path.commonpath([str(root), str(path)]) != str(root):
        raise DetectionDataError("Source path escapes synthetic dataset root.")
    return np.asarray(np.load(path))


def _write_numpy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.replace(path)


def _safe_child(root: Path, child: str) -> Path:
    resolved_root = root.resolve()
    child_path = (resolved_root / child).resolve()
    if os.path.commonpath([str(resolved_root), str(child_path)]) != str(resolved_root):
        raise DetectionDataError("Detection output path escapes configured root.")
    return child_path
