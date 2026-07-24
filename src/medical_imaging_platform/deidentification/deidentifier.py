"""DICOM metadata de-identification."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset

from medical_imaging_platform.deidentification.audit import (
    FileDeidentificationAudit,
    SeriesDeidentificationAudit,
    write_audit,
)
from medical_imaging_platform.deidentification.policy import DeidentificationPolicy
from medical_imaging_platform.deidentification.private_tags import remove_private_tags
from medical_imaging_platform.ingestion.loader import load_dicom
from medical_imaging_platform.ingestion.metadata import detect_identifying_keywords
from medical_imaging_platform.synthetic.manifest import sha256_file
from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class DeidentificationError(MedicalImagingPlatformError):
    """Raised when DICOM de-identification fails."""


UID_KEYWORDS = ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")


def deidentify_series(
    input_paths: list[Path],
    *,
    output_dir: Path,
    audit_path: Path,
    policy: DeidentificationPolicy,
    overwrite: bool = False,
    max_file_size_bytes: int,
) -> SeriesDeidentificationAudit:
    """De-identify one DICOM series into an explicit output directory."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise DeidentificationError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    audits: list[FileDeidentificationAudit] = []
    uid_map: dict[str, str] = {}
    source_key = "|".join(str(path) for path in sorted(input_paths))
    research_subject_id = f"{policy.patient_id_prefix}-{_stable_hash(source_key)[:12]}"
    for index, input_path in enumerate(sorted(input_paths)):
        dataset = load_dicom(input_path, header_only=False, max_file_size_bytes=max_file_size_bytes)
        output_path = _safe_output_path(output_dir, f"deidentified-{index + 1:03d}.dcm")
        audits.append(
            deidentify_file(
                dataset,
                source_path=input_path,
                output_path=output_path,
                policy=policy,
                research_subject_id=research_subject_id,
                uid_map=uid_map,
                overwrite=overwrite,
            )
        )
    series_audit = SeriesDeidentificationAudit(
        policy_version=policy.policy_version,
        file_count=len(audits),
        private_tags_removed=sum(audit.private_tags_removed for audit in audits),
        files=audits,
        warnings=sorted({warning for audit in audits for warning in audit.warnings}),
    )
    write_audit(audit_path, series_audit)
    return series_audit


def deidentify_file(
    dataset: FileDataset,
    *,
    source_path: Path,
    output_path: Path,
    policy: DeidentificationPolicy,
    research_subject_id: str,
    uid_map: dict[str, str],
    overwrite: bool = False,
) -> FileDeidentificationAudit:
    """De-identify one loaded DICOM dataset and write the output."""
    if source_path.resolve() == output_path.resolve():
        raise DeidentificationError("Refusing in-place source overwrite")
    if output_path.exists() and not overwrite:
        raise DeidentificationError(f"Output file exists: {output_path}")

    original_tags = detect_identifying_keywords(dataset)
    removed: list[str] = []
    replaced: list[str] = []
    warnings: list[str] = []

    for keyword, action in policy.direct_identifier_actions.items():
        if keyword not in dataset:
            continue
        if action == "REMOVE":
            delattr(dataset, keyword)
            removed.append(keyword)
        elif action == "REPLACE":
            setattr(dataset, keyword, research_subject_id)
            replaced.append(keyword)
        elif action == "HASH":
            setattr(dataset, keyword, _stable_hash(str(getattr(dataset, keyword)))[:16])
            replaced.append(keyword)
        elif action == "KEEP_WITH_JUSTIFICATION":
            warnings.append(f"{keyword} kept by policy justification")

    remapped: list[str] = []
    for keyword in UID_KEYWORDS:
        if keyword in dataset:
            original_uid = str(getattr(dataset, keyword))
            setattr(dataset, keyword, _remap_uid(original_uid, policy.uid_root, uid_map))
            remapped.append(keyword)
    if "MediaStorageSOPInstanceUID" in dataset.file_meta and "SOPInstanceUID" in dataset:
        dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID

    private_removed = remove_private_tags(dataset)
    burned = str(getattr(dataset, "BurnedInAnnotation", "UNKNOWN")).upper()
    if burned == "YES":
        warnings.append(
            "BurnedInAnnotation is YES; metadata de-identification does not remove "
            "pixel identifiers"
        )
    elif burned not in {"NO", "YES"}:
        warnings.append("BurnedInAnnotation is missing or unknown")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataset.save_as(tmp_path, enforce_file_format=True)
    pydicom.dcmread(tmp_path, stop_before_pixels=True, force=False)
    tmp_path.replace(output_path)

    return FileDeidentificationAudit(
        audit_id=f"audit-{_stable_hash(str(source_path))[:16]}",
        source_file_hash=sha256_file(source_path),
        output_file_hash=sha256_file(output_path),
        policy_version=policy.policy_version,
        deidentified_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
        research_subject_id=research_subject_id,
        original_tags_detected=original_tags,
        tags_removed=sorted(removed),
        tags_replaced=sorted(replaced),
        private_tags_removed=private_removed,
        uids_remapped=sorted(remapped),
        burned_in_annotation_status=burned,
        warnings=warnings,
    )


def _remap_uid(source_uid: str, uid_root: str, uid_map: dict[str, str]) -> str:
    if source_uid not in uid_map:
        digest = int(_stable_hash(source_uid)[:30], 16)
        uid_map[source_uid] = f"{uid_root}.{digest}"
    return uid_map[source_uid][:64].rstrip(".")


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_output_path(output_dir: Path, filename: str) -> Path:
    output_root = output_dir.resolve()
    output_path = (output_dir / filename).resolve()
    if output_root not in output_path.parents and output_root != output_path.parent:
        raise DeidentificationError("Output path escapes destination directory")
    return output_path
