"""Synthetic segmentation dataset preparation and loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from medical_imaging_platform.segmentation.models import (
    SegmentationConfig,
    SegmentationDatasetManifest,
    SegmentationSample,
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


class SegmentationDataError(ValueError):
    """Raised when segmentation dataset preparation or loading fails."""


def prepare_segmentation_dataset(
    synthetic_dataset_dir: Path,
    *,
    output_root: Path,
    config: SegmentationConfig,
    overwrite: bool,
) -> SegmentationDatasetManifest:
    """Create segmentation-ready samples from an existing synthetic dataset."""
    source_root = synthetic_dataset_dir.resolve()
    manifest = load_manifest(source_root / MANIFEST_FILENAME)
    splits = DatasetSplits.model_validate_json(
        (source_root / SPLIT_FILENAME).read_text(encoding="utf-8")
    )
    case_to_split = {
        case_id: split_name
        for split_name, case_ids in splits.splits.items()
        for case_id in case_ids
    }
    detect_subject_leakage(
        splits.splits,
        {record.case_id: record.research_subject_id for record in manifest.records},
    )
    dataset_id = f"segmentation-{manifest.dataset_id}-{config.policy_version}"
    root = _safe_child(output_root, dataset_id)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise SegmentationDataError(f"Segmentation dataset already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)

    samples: list[SegmentationSample] = []
    for record in sorted(manifest.records, key=lambda item: item.case_id):
        split = case_to_split.get(record.case_id)
        if split not in {"train", "validation", "test"}:
            raise SegmentationDataError(f"Case missing from split file: {record.case_id}")
        for timepoint in ("previous", "current"):
            image_key = f"{timepoint}_volume"
            mask_key = f"{timepoint}_lesion_mask"
            image = _load_source_array(source_root, record.file_paths[image_key])
            mask = _load_source_array(source_root, record.file_paths[mask_key])
            image, mask = _normalise_pair(image, mask, config.input_shape)
            sample_id = f"{record.case_id}-{timepoint}"
            sample_dir = root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            _write_numpy(sample_dir / "image.npy", image)
            _write_numpy(sample_dir / "lesion_mask.npy", mask.astype(np.uint8))
            samples.append(
                SegmentationSample(
                    sample_id=sample_id,
                    image=str(Path(sample_id) / "image.npy"),
                    lesion_mask=str(Path(sample_id) / "lesion_mask.npy"),
                    case_id=record.case_id,
                    research_subject_id=record.research_subject_id,
                    split=cast(Any, split),
                    scenario=record.scenario,
                    timepoint=cast(Any, timepoint),
                    spacing_mm=record.voxel_spacing_mm,
                    source_checksums={
                        image_key: record.checksums[image_key],
                        mask_key: record.checksums[mask_key],
                    },
                    localisation_run_id=None,
                    lesion_volume_voxels=int(np.count_nonzero(mask)),
                )
            )
    sample_checksums = {
        sample.sample_id: _sample_checksum(root, sample)
        for sample in sorted(samples, key=lambda item: item.sample_id)
    }
    prepared = SegmentationDatasetManifest(
        dataset_id=dataset_id,
        source_dataset_id=manifest.dataset_id,
        policy_version=config.policy_version,
        generated_at=stable_timestamp(),
        input_shape=config.input_shape,
        samples=sorted(samples, key=lambda item: item.sample_id),
        split_counts={
            split_name: sum(1 for sample in samples if sample.split == split_name)
            for split_name in ("train", "validation", "test")
        },
        checksums=sample_checksums,
    )
    write_json(root / "segmentation_manifest.json", prepared.model_dump(mode="json"))
    return validate_segmentation_dataset(root)


def validate_segmentation_dataset(dataset_dir: Path) -> SegmentationDatasetManifest:
    """Validate a prepared segmentation dataset and checksums."""
    manifest_path = dataset_dir / "segmentation_manifest.json"
    if not manifest_path.exists():
        raise SegmentationDataError("Segmentation dataset manifest is missing.")
    manifest = SegmentationDatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    split_subjects: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for sample in manifest.samples:
        image = np.load(dataset_dir / sample.image)
        mask = np.load(dataset_dir / sample.lesion_mask)
        _validate_sample_arrays(image, mask, manifest.input_shape)
        expected = manifest.checksums[sample.sample_id]
        if _sample_checksum(dataset_dir, sample) != expected:
            raise SegmentationDataError(f"Checksum mismatch for sample {sample.sample_id}")
        split_subjects[sample.split].append(sample.research_subject_id)
    for first_name, first_subjects in split_subjects.items():
        for second_name, second_subjects in split_subjects.items():
            if first_name < second_name and set(first_subjects) & set(second_subjects):
                raise SegmentationDataError("Research subject leakage detected.")
    return manifest


class SegmentationTorchDataset:
    """PyTorch dataset for prepared segmentation samples."""

    def __init__(
        self,
        dataset_dir: Path,
        manifest: SegmentationDatasetManifest,
        split: str,
        transform: Any | None = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.samples = [sample for sample in manifest.samples if sample.split == split]
        self.transform = transform
        if not self.samples:
            raise SegmentationDataError(f"No samples available for split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        item: dict[str, Any] = {
            "image": np.load(self.dataset_dir / sample.image).astype(np.float32)[None],
            "mask": np.load(self.dataset_dir / sample.lesion_mask).astype(np.float32)[None],
            "sample_id": sample.sample_id,
            "spacing_mm": np.asarray(sample.spacing_mm, dtype=np.float32),
            "scenario": sample.scenario,
        }
        if self.transform is not None:
            item = self.transform(item)
        item["image"] = torch.as_tensor(item["image"], dtype=torch.float32)
        item["mask"] = torch.as_tensor(item["mask"], dtype=torch.float32)
        return item


def _normalise_pair(
    image: np.ndarray, mask: np.ndarray, input_shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    _validate_sample_arrays(image, mask, input_shape)
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise SegmentationDataError("Image contains no finite voxels.")
    lower = float(np.percentile(finite, 1))
    upper = float(np.percentile(finite, 99))
    if upper <= lower:
        normalised = np.zeros_like(image, dtype=np.float32)
    else:
        normalised = np.clip((image - lower) / (upper - lower), 0.0, 1.0)
    return normalised.astype(np.float32), mask.astype(bool)


def _validate_sample_arrays(
    image: np.ndarray, mask: np.ndarray, input_shape: tuple[int, int, int]
) -> None:
    if image.shape != input_shape or mask.shape != input_shape:
        raise SegmentationDataError("Image and mask must match configured input_shape.")
    if image.ndim != 3 or mask.ndim != 3:
        raise SegmentationDataError("Image and mask must be 3D arrays.")
    if not np.all(np.isfinite(image)):
        raise SegmentationDataError("Image contains non-finite values.")
    if not set(np.unique(mask)).issubset({0, 1}):
        raise SegmentationDataError("Lesion mask must be binary.")


def _load_source_array(root: Path, relative_path: str) -> np.ndarray:
    path = (root / relative_path).resolve()
    if os.path.commonpath([str(root), str(path)]) != str(root):
        raise SegmentationDataError("Synthetic source path escapes dataset root.")
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
        raise SegmentationDataError("Segmentation output path escapes configured root.")
    return child_path


def _sample_checksum(root: Path, sample: SegmentationSample) -> str:
    return sha256_file(root / sample.image) + sha256_file(root / sample.lesion_mask)
