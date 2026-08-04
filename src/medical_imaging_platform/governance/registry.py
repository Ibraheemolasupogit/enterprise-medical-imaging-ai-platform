"""Local governed model registry evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from medical_imaging_platform.governance.models import (
    ApprovalMetadata,
    ModelType,
    ModelVersionRecord,
    RegistryManifest,
)
from medical_imaging_platform.release.checksums import checksum_paths

REGISTRY_DIR = Path("reports/generated/registry")
REGISTRY_PATH = REGISTRY_DIR / "registry_manifest.json"
REGISTRY_REPORT = REGISTRY_DIR / "registry_report.md"
REGISTRY_CHECKSUMS = REGISTRY_DIR / "checksum_manifest.json"


def file_or_reference_checksum(reference: str) -> str:
    """Return a SHA-256 checksum for a local file or deterministic reference string."""
    path = Path(reference)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(reference.encode("utf-8")).hexdigest()


def load_registry(path: Path = REGISTRY_PATH) -> RegistryManifest:
    """Load registry manifest or return an empty one."""
    if not path.is_file():
        return RegistryManifest()
    return RegistryManifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_registry(manifest: RegistryManifest, path: Path = REGISTRY_PATH) -> Path:
    """Persist registry manifest and companion evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report_path = path.parent / "registry_report.md"
    checksum_path = path.parent / "checksum_manifest.json"
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(registry_report(manifest), encoding="utf-8")
    checksum_path.write_text(
        json.dumps(checksum_paths([path, report_path]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def build_model_record(
    *,
    model_type: ModelType,
    version: str,
    checkpoint_reference: str,
    config_reference: str,
    training_data_reference: str,
) -> ModelVersionRecord:
    """Build a deterministic candidate registry record."""
    metrics: dict[str, float] = (
        {"dice": 0.74, "recall": 0.82, "false_positive_voxel_rate": 0.04}
        if model_type == "segmentation"
        else {"auroc": 0.86, "brier_score": 0.14, "expected_calibration_error": 0.06}
    )
    thresholds: dict[str, float] = (
        {"mask_probability_threshold": 0.50, "minimum_component_voxels": 8.0}
        if model_type == "segmentation"
        else {"positive_threshold": 0.62, "abstain_low": 0.45, "abstain_high": 0.55}
    )
    calibration: dict[str, Any] = (
        {"method": "not_applicable_for_synthetic_segmentation_baseline"}
        if model_type == "segmentation"
        else {"method": "validation_only_temperature_scaling", "temperature": 1.15}
    )
    return ModelVersionRecord(
        model_name=f"synthetic-{model_type}-baseline",
        version=version,
        model_type=model_type,
        checkpoint_reference=checkpoint_reference,
        checkpoint_checksum=file_or_reference_checksum(checkpoint_reference),
        framework_versions={
            "python": "3.12",
            "torch": "2.13.0+cpu",
            "monai": "1.6.0",
            "numpy": "1.26",
        },
        config_reference=config_reference,
        config_checksum=file_or_reference_checksum(config_reference),
        training_data_reference=training_data_reference,
        evaluation_metrics=metrics,
        thresholds=thresholds,
        calibration_metadata=calibration,
    )


def register_model(record: ModelVersionRecord, path: Path = REGISTRY_PATH) -> RegistryManifest:
    """Register one candidate model version without automatic promotion."""
    manifest = load_registry(path)
    if any(
        item.model_name == record.model_name and item.version == record.version
        for item in manifest.records
    ):
        raise ValueError(f"Duplicate model version: {record.model_name}:{record.version}")
    manifest.records.append(record)
    manifest.records.sort(key=lambda item: (item.model_name, item.version))
    save_registry(manifest, path)
    return manifest


def approve_model(
    model_name: str,
    version: str,
    approval: ApprovalMetadata,
    path: Path = REGISTRY_PATH,
) -> RegistryManifest:
    """Explicitly approve one registered candidate model version."""
    manifest = load_registry(path)
    updated: list[ModelVersionRecord] = []
    found = False
    for record in manifest.records:
        if record.model_name == model_name and record.version == version:
            found = True
            updated.append(
                record.model_copy(
                    update={"lifecycle_state": "approved", "approval_metadata": approval}
                )
            )
        else:
            updated.append(record)
    if not found:
        raise ValueError(f"Model version not found: {model_name}:{version}")
    manifest = manifest.model_copy(update={"records": updated})
    save_registry(manifest, path)
    return manifest


def set_lifecycle_state(
    model_name: str,
    version: str,
    state: str,
    path: Path = REGISTRY_PATH,
) -> RegistryManifest:
    """Set a non-approved lifecycle state for local governance evidence."""
    if state == "approved":
        raise ValueError("Use approve_model with explicit approval metadata.")
    manifest = load_registry(path)
    updated: list[ModelVersionRecord] = []
    found = False
    for record in manifest.records:
        if record.model_name == model_name and record.version == version:
            found = True
            updated.append(record.model_copy(update={"lifecycle_state": state}))
        else:
            updated.append(record)
    if not found:
        raise ValueError(f"Model version not found: {model_name}:{version}")
    manifest = manifest.model_copy(update={"records": updated})
    save_registry(manifest, path)
    return manifest


def ensure_demo_registry(path: Path = REGISTRY_PATH) -> RegistryManifest:
    """Create deterministic segmentation and classification registry records if absent."""
    manifest = load_registry(path)
    existing = {(record.model_type, record.version) for record in manifest.records}
    defaults: list[tuple[ModelType, str, str]] = [
        ("segmentation", "m14-segmentation-synthetic-v1", "config/segmentation.yaml"),
        ("classification", "m14-classification-synthetic-v1", "config/classification.yaml"),
    ]
    for model_type, version, config_ref in defaults:
        if (model_type, version) not in existing:
            manifest.records.append(
                build_model_record(
                    model_type=model_type,
                    version=version,
                    checkpoint_reference=f"synthetic://{model_type}/best_model.pt",
                    config_reference=config_ref,
                    training_data_reference=f"synthetic://m14/{model_type}/dataset",
                )
            )
    manifest.records.sort(key=lambda item: (item.model_name, item.version))
    save_registry(manifest, path)
    return manifest


def registry_report(manifest: RegistryManifest) -> str:
    """Render local registry Markdown evidence."""
    lines = [
        "# Governed Model Registry Evidence",
        "",
        manifest.disclaimer,
        "",
        "| Model | Version | Type | State | Approved |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in manifest.records:
        approved = "yes" if record.approval_metadata is not None else "no"
        lines.append(
            f"| {record.model_name} | {record.version} | {record.model_type} | "
            f"{record.lifecycle_state} | {approved} |"
        )
    lines.extend(
        [
            "",
            "Lifecycle transitions are human-governed. Registration never promotes a model "
            "automatically.",
            "",
        ]
    )
    return "\n".join(lines)
