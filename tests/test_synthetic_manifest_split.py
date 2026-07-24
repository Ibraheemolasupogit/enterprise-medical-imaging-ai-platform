from pathlib import Path

import numpy as np
import pytest

from medical_imaging_platform.synthetic.generator import SyntheticDataConfig, generate_cases
from medical_imaging_platform.synthetic.io import (
    SyntheticDataError,
    generate_dataset,
    validate_dataset,
)
from medical_imaging_platform.synthetic.manifest import (
    load_manifest,
    sha256_file,
    write_case,
    write_manifest,
)
from medical_imaging_platform.synthetic.split import create_subject_splits
from medical_imaging_platform.synthetic.validation import (
    DatasetValidationError,
    detect_subject_leakage,
)


def config(tmp_path: Path) -> SyntheticDataConfig:
    return SyntheticDataConfig(
        dataset_id="test-synthetic",
        dataset_version="0.2.0",
        manifest_version="1.0",
        output_root=tmp_path,
        dataset_size=6,
        random_seed=101,
        volume_shape=(32, 32, 32),
        voxel_spacing_mm=(2.5, 2.5, 2.5),
        noise_std_hu=1.0,
        adrenal_radius_voxels=4,
        lesion_radius_voxels=2,
        lesion_intensity_hu=95.0,
        scenarios=["stable", "increased", "reduced", "new", "resolved", "translated"],
        split_ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
        generator_version="test-generator",
    )


def test_manifest_validation_and_checksums(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    manifest = generate_dataset(cfg, tmp_path, overwrite=True)

    validated = validate_dataset(tmp_path)

    assert validated == manifest
    assert len(validated.records[0].checksums["metadata"]) == 64


def test_checksum_generation(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("same content\n", encoding="utf-8")

    assert sha256_file(file_path) == sha256_file(file_path)


def test_stable_manifest_serialisation(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    cases = generate_cases(cfg, 2)
    records = [write_case(case, cfg, tmp_path) for case in reversed(cases)]

    first = write_manifest(records, cfg, tmp_path)
    first_text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    second = write_manifest(list(reversed(records)), cfg, tmp_path)
    second_text = (tmp_path / "manifest.json").read_text(encoding="utf-8")

    assert first == second
    assert first_text == second_text


def test_dataset_summary(tmp_path: Path) -> None:
    manifest = generate_dataset(config(tmp_path), tmp_path, overwrite=True)

    assert manifest.summary["case_count"] == 6
    assert manifest.summary["research_subject_count"] == 6
    assert manifest.summary["scenario_counts"]["stable"] == 1


def test_deterministic_subject_level_splitting(tmp_path: Path) -> None:
    manifest = generate_dataset(config(tmp_path), tmp_path, overwrite=True)

    first = create_subject_splits(
        manifest,
        {"train": 0.5, "validation": 0.25, "test": 0.25},
        seed=99,
    )
    second = create_subject_splits(
        manifest,
        {"train": 0.5, "validation": 0.25, "test": 0.25},
        seed=99,
    )

    assert first == second
    assert sorted(first.splits) == ["test", "train", "validation"]


def test_no_subject_leakage_detection() -> None:
    with pytest.raises(DatasetValidationError, match="appears in both"):
        detect_subject_leakage(
            {"train": ["case-1"], "validation": ["case-2"], "test": []},
            {"case-1": "subject-1", "case-2": "subject-1"},
        )


def test_overwrite_protection(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    tmp_path.joinpath("existing.txt").write_text("already here", encoding="utf-8")

    with pytest.raises(SyntheticDataError, match="not empty"):
        generate_dataset(cfg, tmp_path, overwrite=False)


def test_manifest_checksum_validation_fails_when_file_changes(tmp_path: Path) -> None:
    generate_dataset(config(tmp_path), tmp_path, overwrite=True)
    record = load_manifest(tmp_path / "manifest.json").records[0]
    changed_file = tmp_path / record.file_paths["previous_volume"]
    np.save(changed_file, np.zeros((2, 2), dtype=np.float32))

    with pytest.raises(DatasetValidationError, match="checksum mismatch"):
        validate_dataset(tmp_path)
