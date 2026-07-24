"""Synthetic dataset orchestration helpers."""

from __future__ import annotations

from pathlib import Path

from medical_imaging_platform.synthetic.generator import SyntheticDataConfig, generate_cases
from medical_imaging_platform.synthetic.manifest import (
    MANIFEST_FILENAME,
    DatasetManifest,
    load_manifest,
    validate_manifest_files,
    write_case,
    write_manifest,
)
from medical_imaging_platform.synthetic.split import create_subject_splits, write_splits
from medical_imaging_platform.synthetic.validation import validate_cases
from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class SyntheticDataError(MedicalImagingPlatformError):
    """Raised when synthetic dataset operations fail."""


def generate_dataset(
    config: SyntheticDataConfig,
    output_root: Path,
    case_count: int | None = None,
    random_seed: int | None = None,
    overwrite: bool = False,
) -> DatasetManifest:
    """Generate a complete synthetic dataset with manifest and splits."""
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise SyntheticDataError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    effective_config = config.model_copy(
        update={
            "output_root": output_root,
            "dataset_size": case_count or config.dataset_size,
            "random_seed": config.random_seed if random_seed is None else random_seed,
        }
    )
    cases = generate_cases(effective_config, effective_config.dataset_size)
    validate_cases(cases)

    records = [write_case(case, effective_config, output_root) for case in cases]
    manifest = write_manifest(records, effective_config, output_root)
    splits = create_subject_splits(
        manifest, effective_config.split_ratios, effective_config.random_seed
    )
    write_splits(splits, output_root)
    return manifest


def validate_dataset(dataset_root: Path) -> DatasetManifest:
    """Validate a generated synthetic dataset."""
    manifest = load_manifest(dataset_root / MANIFEST_FILENAME)
    validate_manifest_files(manifest, dataset_root)
    return manifest
