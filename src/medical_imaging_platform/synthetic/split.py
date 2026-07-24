"""Deterministic research-subject-level dataset splitting."""

from __future__ import annotations

import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from medical_imaging_platform.synthetic.manifest import DatasetManifest, write_json
from medical_imaging_platform.synthetic.validation import detect_subject_leakage

SPLIT_FILENAME = "splits.json"


class DatasetSplits(BaseModel):
    """Case IDs assigned to each dataset split."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    splits: dict[str, list[str]]


def create_subject_splits(
    manifest: DatasetManifest,
    split_ratios: dict[str, float],
    seed: int,
) -> DatasetSplits:
    """Create deterministic splits without subject leakage."""
    subject_to_cases: dict[str, list[str]] = {}
    for record in manifest.records:
        subject_to_cases.setdefault(record.research_subject_id, []).append(record.case_id)

    subjects = sorted(subject_to_cases)
    # Deterministic splitting is required for reproducible research fixtures.
    rng = random.Random(seed)  # nosec B311
    rng.shuffle(subjects)

    total = len(subjects)
    train_end = int(total * split_ratios["train"])
    validation_end = train_end + int(total * split_ratios["validation"])
    split_subjects = {
        "train": subjects[:train_end],
        "validation": subjects[train_end:validation_end],
        "test": subjects[validation_end:],
    }
    splits = {
        split_name: sorted(
            case_id for subject in split_subject_list for case_id in subject_to_cases[subject]
        )
        for split_name, split_subject_list in split_subjects.items()
    }
    detect_subject_leakage(
        splits,
        {record.case_id: record.research_subject_id for record in manifest.records},
    )
    return DatasetSplits(seed=seed, splits=splits)


def write_splits(splits: DatasetSplits, output_root: Path) -> None:
    """Write deterministic split JSON."""
    write_json(output_root / SPLIT_FILENAME, splits.model_dump(mode="json"))
