"""De-identification audit records."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class FileDeidentificationAudit(BaseModel):
    """One-file de-identification audit without direct identifier values."""

    model_config = ConfigDict(extra="forbid")

    audit_id: str
    source_file_hash: str
    output_file_hash: str
    policy_version: str
    deidentified_at: str
    research_subject_id: str
    original_tags_detected: list[str]
    tags_removed: list[str]
    tags_replaced: list[str]
    private_tags_removed: int
    uids_remapped: list[str]
    burned_in_annotation_status: str
    warnings: list[str]


class SeriesDeidentificationAudit(BaseModel):
    """Series-level de-identification summary."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str
    file_count: int
    private_tags_removed: int
    files: list[FileDeidentificationAudit]
    warnings: list[str]


def write_audit(path: Path, audit: SeriesDeidentificationAudit) -> None:
    """Write deterministic audit JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
