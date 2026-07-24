"""Manifest generation, checksums, serialisation, and summaries."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from medical_imaging_platform.synthetic.generator import SyntheticCase, SyntheticDataConfig
from medical_imaging_platform.synthetic.validation import DatasetValidationError

MANIFEST_FILENAME = "manifest.json"


class ManifestRecord(BaseModel):
    """One synthetic longitudinal case manifest record."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_version: str
    case_id: str
    research_subject_id: str
    pair_status: str
    previous_study_id: str
    current_study_id: str
    scenario: str
    volume_shape: tuple[int, int, int]
    voxel_spacing_mm: tuple[float, float, float]
    lesion_side: str
    previous_lesion_volume_voxels: int = Field(ge=0)
    current_lesion_volume_voxels: int = Field(ge=0)
    random_seed: int
    generator_version: str
    generation_timestamp: str
    file_paths: dict[str, str]
    checksums: dict[str, str]
    change_metadata: dict[str, Any]


class DatasetManifest(BaseModel):
    """Dataset-level manifest with records and summary statistics."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: str
    dataset_id: str
    dataset_version: str
    generated_at: str
    records: list[ManifestRecord]
    summary: dict[str, Any]


def write_case(
    case: SyntheticCase, config: SyntheticDataConfig, output_root: Path
) -> ManifestRecord:
    """Write one case directory and return its manifest record."""
    case_dir = output_root / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "previous_volume": case.previous_volume,
        "current_volume": case.current_volume,
        "previous_lesion_mask": case.previous_lesion_mask.astype(np.uint8),
        "current_lesion_mask": case.current_lesion_mask.astype(np.uint8),
        "previous_left_adrenal_mask": case.previous_left_adrenal_mask.astype(np.uint8),
        "previous_right_adrenal_mask": case.previous_right_adrenal_mask.astype(np.uint8),
        "current_left_adrenal_mask": case.current_left_adrenal_mask.astype(np.uint8),
        "current_right_adrenal_mask": case.current_right_adrenal_mask.astype(np.uint8),
    }
    file_paths: dict[str, str] = {}
    checksums: dict[str, str] = {}
    for name, array in arrays.items():
        file_path = case_dir / f"{name}.npy"
        np.save(file_path, array)
        relative_path = str(file_path.relative_to(output_root))
        file_paths[name] = relative_path
        checksums[name] = sha256_file(file_path)

    metadata_path = case_dir / "metadata.json"
    metadata = {
        "case_id": case.case_id,
        "research_subject_id": case.research_subject_id,
        "scenario": case.scenario,
        "lesion_side": case.lesion_side,
        "change_metadata": case.change_metadata,
    }
    write_json(metadata_path, metadata)
    file_paths["metadata"] = str(metadata_path.relative_to(output_root))
    checksums["metadata"] = sha256_file(metadata_path)

    return ManifestRecord(
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        case_id=case.case_id,
        research_subject_id=case.research_subject_id,
        pair_status="paired",
        previous_study_id=f"{case.case_id}-previous",
        current_study_id=f"{case.case_id}-current",
        scenario=case.scenario,
        volume_shape=config.volume_shape,
        voxel_spacing_mm=config.voxel_spacing_mm,
        lesion_side=case.lesion_side,
        previous_lesion_volume_voxels=int(np.sum(case.previous_lesion_mask)),
        current_lesion_volume_voxels=int(np.sum(case.current_lesion_mask)),
        random_seed=case.random_seed,
        generator_version=config.generator_version,
        generation_timestamp=stable_timestamp(),
        file_paths=file_paths,
        checksums=checksums,
        change_metadata=case.change_metadata,
    )


def write_manifest(
    records: list[ManifestRecord],
    config: SyntheticDataConfig,
    output_root: Path,
) -> DatasetManifest:
    """Write a deterministic dataset manifest."""
    sorted_records = sorted(records, key=lambda record: record.case_id)
    manifest = DatasetManifest(
        manifest_version=config.manifest_version,
        dataset_id=config.dataset_id,
        dataset_version=config.dataset_version,
        generated_at=stable_timestamp(),
        records=sorted_records,
        summary=summarise_records(sorted_records),
    )
    write_json(output_root / MANIFEST_FILENAME, manifest.model_dump(mode="json"))
    return manifest


def load_manifest(path: Path) -> DatasetManifest:
    """Load and validate a dataset manifest."""
    if not path.exists():
        raise DatasetValidationError(f"Manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DatasetManifest.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise DatasetValidationError(f"Invalid manifest {path}: {exc}") from exc


def validate_manifest_files(manifest: DatasetManifest, dataset_root: Path) -> None:
    """Validate manifest file paths and checksums."""
    for record in manifest.records:
        for name, relative_path in record.file_paths.items():
            file_path = dataset_root / relative_path
            if not file_path.exists():
                raise DatasetValidationError(
                    f"{record.case_id}: missing file for {name}: {relative_path}"
                )
            actual = sha256_file(file_path)
            expected = record.checksums[name]
            if actual != expected:
                raise DatasetValidationError(f"{record.case_id}: checksum mismatch for {name}")


def summarise_records(records: list[ManifestRecord]) -> dict[str, Any]:
    """Create dataset-level summary statistics."""
    scenario_counts: dict[str, int] = {}
    for record in records:
        scenario_counts[record.scenario] = scenario_counts.get(record.scenario, 0) + 1
    return {
        "case_count": len(records),
        "research_subject_count": len({record.research_subject_id for record in records}),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "total_previous_lesion_volume_voxels": sum(
            record.previous_lesion_volume_voxels for record in records
        ),
        "total_current_lesion_volume_voxels": sum(
            record.current_lesion_volume_voxels for record in records
        ),
    }


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stable_timestamp() -> str:
    """Return a deterministic timestamp for reproducible engineering fixtures."""
    return datetime(2026, 1, 1, tzinfo=UTC).isoformat()
