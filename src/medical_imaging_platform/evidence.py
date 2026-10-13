"""Small additive evidence contract; existing domain manifests remain authoritative."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def generated_timestamp() -> str:
    return datetime.now(UTC).isoformat()


CapabilityStatus = Literal[
    "implemented", "executed/validated", "simulated", "static design", "target state"
]


def array_checksum(array: np.ndarray) -> str:
    """Digest shape, dtype and contiguous bytes; distinct from a .npy file digest."""
    header = json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True)
    return hashlib.sha256(header.encode() + np.ascontiguousarray(array).tobytes()).hexdigest()


class ArtifactBinding(BaseModel):
    """Exact artifact identity carried from model response into human review."""

    model_config = ConfigDict(extra="forbid")
    artifact_type: str = Field(min_length=1)
    artifact_reference: str = Field(min_length=1)
    artifact_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    checksum_kind: Literal["sha256-file", "sha256-array"] = "sha256-file"
    run_id: str = Field(min_length=1)
    source_checksum: str | None = None
    model_family: str | None = None
    checkpoint_checksum: str | None = None


def verify_artifact(binding: ArtifactBinding, path: Path) -> None:
    actual = (
        sha256_file(path)
        if binding.checksum_kind == "sha256-file"
        else array_checksum(np.load(path, allow_pickle=False))
    )
    if actual != binding.artifact_checksum:
        raise ValueError("Reviewed artifact checksum mismatch")


class EvidenceEnvelope(BaseModel):
    """Sidecar metadata, never a replacement for task-specific evidence."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "M23-v1"
    artifact_type: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    artifact_reference: str = Field(min_length=1)
    artifact_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_id: str | None = None
    source_checksum: str | None = None
    model_family: str | None = None
    checkpoint_checksum: str | None = None
    config_checksum: str | None = None
    generated_at: str = Field(default_factory=generated_timestamp)
    quality_status: str
    capability_status: CapabilityStatus = "executed/validated"
    data_role: Literal["predicted", "reference", "derived", "simulated", "static"] = "derived"
    warnings: list[str] = Field(default_factory=list)


def evidence_checksum(payload: dict[str, Any]) -> str:
    """Hash canonical metadata; exclude only its own digest field."""
    content = {key: value for key, value in payload.items() if key != "evidence_checksum"}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_envelope(path: Path, envelope: EvidenceEnvelope) -> dict[str, Any]:
    payload = envelope.model_dump(mode="json")
    payload["evidence_checksum"] = evidence_checksum(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_envelope(path: Path, artifact: Path) -> EvidenceEnvelope:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.pop("evidence_checksum", None) != evidence_checksum(payload):
        raise ValueError("Evidence envelope checksum mismatch")
    envelope = EvidenceEnvelope.model_validate(payload)
    if sha256_file(artifact) != envelope.artifact_checksum:
        raise ValueError("Reviewed artifact checksum mismatch")
    return envelope
