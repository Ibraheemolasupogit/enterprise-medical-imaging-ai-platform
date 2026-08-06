"""Classification dataset preparation from synthetic ROI-like crops."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from medical_imaging_platform.classification.models import (
    ClassificationConfig,
    ClassificationDatasetManifest,
    ClassificationSample,
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


class ClassificationDataError(ValueError):
    """Raised when classification data are invalid."""


def prepare_classification_dataset(
    synthetic_dataset_dir: Path,
    *,
    output_root: Path,
    config: ClassificationConfig,
    overwrite: bool,
) -> ClassificationDatasetManifest:
    """Create deterministic positive/negative ROI crops from synthetic data."""
    source_root = synthetic_dataset_dir.resolve()
    manifest = load_manifest(source_root / MANIFEST_FILENAME)
    splits = DatasetSplits.model_validate_json(
        (source_root / SPLIT_FILENAME).read_text(encoding="utf-8")
    )
    detect_subject_leakage(
        splits.splits,
        {record.case_id: record.research_subject_id for record in manifest.records},
    )
    case_label_sets = _case_label_sets(source_root, manifest.records)
    case_to_split = _classification_case_splits(splits.splits, case_label_sets, config.random_seed)
    dataset_id = f"classification-{manifest.dataset_id}-{config.policy_version}"
    root = _safe_child(output_root, dataset_id)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise ClassificationDataError(f"Classification dataset already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)

    samples: list[ClassificationSample] = []
    for record in sorted(manifest.records, key=lambda item: item.case_id):
        split = case_to_split.get(record.case_id)
        if split not in {"train", "validation", "test"}:
            raise ClassificationDataError(f"Case missing from split file: {record.case_id}")
        for timepoint in ("previous", "current"):
            volume_key = f"{timepoint}_volume"
            mask_key = f"{timepoint}_lesion_mask"
            adrenal_key = f"{timepoint}_{record.lesion_side}_adrenal_mask"
            image = _load_array(source_root, record.file_paths[volume_key])
            lesion_mask = _load_array(source_root, record.file_paths[mask_key]).astype(bool)
            adrenal_mask = _load_array(source_root, record.file_paths[adrenal_key]).astype(bool)
            crop = _crop_to_mask(image, adrenal_mask, config.input_shape)
            crop = _normalise(crop)
            label = int(np.any(lesion_mask & adrenal_mask))
            sample_id = f"{record.case_id}-{timepoint}-{record.lesion_side}"
            sample_dir = root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            _write_numpy(sample_dir / "image.npy", crop)
            samples.append(
                ClassificationSample(
                    sample_id=sample_id,
                    image=str(Path(sample_id) / "image.npy"),
                    label=cast(Any, label),
                    label_name=(
                        "synthetic_lesion_present" if label == 1 else "no_visible_synthetic_lesion"
                    ),
                    case_id=record.case_id,
                    research_subject_id=record.research_subject_id,
                    split=cast(Any, split),
                    scenario=record.scenario,
                    side=cast(Any, record.lesion_side),
                    timepoint=cast(Any, timepoint),
                    spacing_mm=record.voxel_spacing_mm,
                    source_checksums={
                        volume_key: record.checksums[volume_key],
                        mask_key: record.checksums[mask_key],
                        adrenal_key: record.checksums[adrenal_key],
                    },
                    localisation_run_id=None,
                    segmentation_run_id=None,
                )
            )
    prepared = ClassificationDatasetManifest(
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
        class_counts={
            "no_visible_synthetic_lesion": sum(1 for sample in samples if sample.label == 0),
            "synthetic_lesion_present": sum(1 for sample in samples if sample.label == 1),
        },
        checksums={sample.sample_id: sha256_file(root / sample.image) for sample in samples},
        label_semantics={
            "0": "no_visible_synthetic_lesion",
            "1": "synthetic_lesion_present",
            "indeterminate": "inference-only abstention label, not a training class",
        },
    )
    write_json(root / "classification_manifest.json", prepared.model_dump(mode="json"))
    return validate_classification_dataset(root)


def validate_classification_dataset(dataset_dir: Path) -> ClassificationDatasetManifest:
    """Validate classification dataset files, splits, labels, and checksums."""
    manifest_path = dataset_dir / "classification_manifest.json"
    if not manifest_path.exists():
        raise ClassificationDataError("Classification manifest is missing.")
    manifest = ClassificationDatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    split_subjects: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    split_labels: dict[str, set[int]] = {"train": set(), "validation": set(), "test": set()}
    for sample in manifest.samples:
        image = np.load(dataset_dir / sample.image)
        _validate_image(image, manifest.input_shape)
        if sample.label not in {0, 1}:
            raise ClassificationDataError("Classification label must be binary.")
        if sha256_file(dataset_dir / sample.image) != manifest.checksums[sample.sample_id]:
            raise ClassificationDataError(f"Checksum mismatch for sample {sample.sample_id}")
        split_subjects[sample.split].add(sample.research_subject_id)
        split_labels[sample.split].add(sample.label)
    for first_name, first_subjects in split_subjects.items():
        for second_name, second_subjects in split_subjects.items():
            if first_name < second_name and first_subjects & second_subjects:
                raise ClassificationDataError("Research subject leakage detected.")
    for split_name, labels in split_labels.items():
        if labels != {0, 1}:
            raise ClassificationDataError(
                f"Split must contain positive and negative cases: {split_name}"
            )
    return manifest


def _case_label_sets(root: Path, records: list[Any]) -> dict[str, set[int]]:
    case_labels: dict[str, set[int]] = {}
    for record in records:
        labels = set()
        for timepoint in ("previous", "current"):
            mask_key = f"{timepoint}_lesion_mask"
            adrenal_key = f"{timepoint}_{record.lesion_side}_adrenal_mask"
            lesion_mask = _load_array(root, record.file_paths[mask_key]).astype(bool)
            adrenal_mask = _load_array(root, record.file_paths[adrenal_key]).astype(bool)
            labels.add(int(np.any(lesion_mask & adrenal_mask)))
        case_labels[record.case_id] = labels
    return case_labels


def _classification_case_splits(
    source_splits: dict[str, list[str]], case_label_sets: dict[str, set[int]], seed: int
) -> dict[str, str]:
    if _splits_are_binary(source_splits, case_label_sets):
        return {
            case_id: split_name
            for split_name, case_ids in source_splits.items()
            for case_id in case_ids
        }
    split_names = ("train", "validation", "test")
    mixed = sorted(case_id for case_id, labels in case_label_sets.items() if labels == {0, 1})
    if len(mixed) < len(split_names):
        raise ClassificationDataError(
            "Classification split cannot contain both labels in every split."
        )
    all_cases = sorted(case_label_sets)
    # Deterministic engineering split only.
    rng = random.Random(seed)  # nosec B311
    rng.shuffle(mixed)
    remaining = [case_id for case_id in all_cases if case_id not in set(mixed)]
    rng.shuffle(remaining)
    ordered_remaining = mixed[len(split_names) :] + remaining
    targets = {
        split_name: max(1, len(source_splits.get(split_name, []))) for split_name in split_names
    }
    assignments: dict[str, list[str]] = {
        split_name: [mixed[index]] for index, split_name in enumerate(split_names)
    }
    for case_id in ordered_remaining:
        target_split = min(
            split_names,
            key=lambda split_name: (
                len(assignments[split_name]) / targets[split_name],
                len(assignments[split_name]),
                split_name,
            ),
        )
        assignments[target_split].append(case_id)
    if not _splits_are_binary(assignments, case_label_sets):
        raise ClassificationDataError(
            "Classification split cannot contain both labels in every split."
        )
    return {
        case_id: split_name
        for split_name, case_ids in assignments.items()
        for case_id in sorted(case_ids)
    }


def _splits_are_binary(splits: dict[str, list[str]], case_label_sets: dict[str, set[int]]) -> bool:
    for split_name in ("train", "validation", "test"):
        labels: set[int] = set()
        for case_id in splits.get(split_name, []):
            labels.update(case_label_sets.get(case_id, set()))
        if labels != {0, 1}:
            return False
    return True


class ClassificationTorchDataset:
    """Torch-compatible classification dataset."""

    def __init__(
        self,
        dataset_dir: Path,
        manifest: ClassificationDatasetManifest,
        split: str,
        transform: Any | None = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.samples = [sample for sample in manifest.samples if sample.split == split]
        self.transform = transform
        if not self.samples:
            raise ClassificationDataError(f"No samples for split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        item: dict[str, Any] = {
            "image": np.load(self.dataset_dir / sample.image).astype(np.float32)[None],
            "label": np.asarray(sample.label, dtype=np.float32),
            "sample_id": sample.sample_id,
        }
        if self.transform is not None:
            item = self.transform(item)
        item["image"] = torch.as_tensor(item["image"], dtype=torch.float32)
        item["label"] = torch.as_tensor(item["label"], dtype=torch.float32)
        return item


def _crop_to_mask(image: np.ndarray, mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    centre = np.mean(np.argwhere(mask), axis=0) if np.any(mask) else np.array(image.shape) / 2
    starts = [int(round(centre[axis] - shape[axis] / 2)) for axis in range(3)]
    cropped = np.full(shape, float(np.min(image)), dtype=np.float32)
    source_slices = []
    target_slices = []
    for axis, start in enumerate(starts):
        end = start + shape[axis]
        src0 = max(0, start)
        src1 = min(image.shape[axis], end)
        dst0 = max(0, -start)
        dst1 = dst0 + (src1 - src0)
        source_slices.append(slice(src0, src1))
        target_slices.append(slice(dst0, dst1))
    cropped[tuple(target_slices)] = image[tuple(source_slices)]
    return cropped


def _normalise(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ClassificationDataError("Image contains no finite voxels.")
    lower = float(np.percentile(finite, 1))
    upper = float(np.percentile(finite, 99))
    if upper <= lower:
        return np.zeros_like(image, dtype=np.float32)
    clipped = np.clip((image.astype(np.float32) - lower) / (upper - lower), 0.0, 1.0).astype(
        np.float32
    )
    return cast(np.ndarray, clipped)


def _validate_image(image: np.ndarray, shape: tuple[int, int, int]) -> None:
    if image.shape != shape or image.ndim != 3:
        raise ClassificationDataError("Image must match configured 3D input shape.")
    if image.dtype != np.float32:
        raise ClassificationDataError("Classification image must be float32.")
    if not np.all(np.isfinite(image)):
        raise ClassificationDataError("Image contains non-finite values.")


def _load_array(root: Path, relative_path: str) -> np.ndarray:
    path = (root / relative_path).resolve()
    if os.path.commonpath([str(root), str(path)]) != str(root):
        raise ClassificationDataError("Source path escapes synthetic dataset root.")
    return np.asarray(np.load(path))


def _write_numpy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array.astype(np.float32))
    tmp.replace(path)


def _safe_child(root: Path, child: str) -> Path:
    resolved_root = root.resolve()
    child_path = (resolved_root / child).resolve()
    if os.path.commonpath([str(resolved_root), str(child_path)]) != str(resolved_root):
        raise ClassificationDataError("Classification output path escapes configured root.")
    return child_path
