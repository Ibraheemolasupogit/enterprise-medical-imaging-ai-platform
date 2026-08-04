"""Append-only local audit evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from medical_imaging_platform.governance.models import RESEARCH_DISCLAIMER, AuditEvent
from medical_imaging_platform.release.checksums import checksum_paths

AUDIT_DIR = Path("reports/generated/audit")
AUDIT_LOG_PATH = AUDIT_DIR / "audit_log.jsonl"
AUDIT_REPORT_PATH = AUDIT_DIR / "audit_evidence_report.md"
AUDIT_CHECKSUM_PATH = AUDIT_DIR / "checksum_manifest.json"


def output_checksum(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def append_audit_event(event: AuditEvent, path: Path = AUDIT_LOG_PATH) -> Path:
    """Append one validated event to JSONL audit evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n")
    return path


def build_audit_evidence(path: Path = AUDIT_LOG_PATH) -> list[AuditEvent]:
    """Create deterministic synthetic audit evidence."""
    if path.exists():
        path.unlink()
    events = [
        AuditEvent(
            request_id="req-m14-0001",
            correlation_id="corr-m14-0001",
            timestamp="2026-01-01T00:30:00Z",
            model_version="m14-segmentation-synthetic-v1",
            config_version="config/segmentation.yaml",
            input_provenance_reference="synthetic://monitoring/case-0001",
            output_checksum=output_checksum("synthetic-segmentation-output-0001"),
            reviewer_action="accepted",
            actor_type="human_reviewer",
            metadata={"quality_gate": "PASS"},
        ),
        AuditEvent(
            request_id="req-m14-0002",
            correlation_id="corr-m14-0002",
            timestamp="2026-01-01T00:31:00Z",
            model_version="m14-classification-synthetic-v1",
            config_version="config/classification.yaml",
            input_provenance_reference="synthetic://monitoring/case-0002",
            output_checksum=output_checksum("synthetic-classification-output-0002"),
            reviewer_action="indeterminate",
            override_event="quality_gate_block_reviewed",
            export_event="local_evidence_export",
            actor_type="human_reviewer",
            metadata={"abstention": "visible", "payload": "redacted"},
        ),
    ]
    for event in events:
        append_audit_event(event, path)
    AUDIT_REPORT_PATH.write_text(audit_report(events), encoding="utf-8")
    AUDIT_CHECKSUM_PATH.write_text(
        json.dumps(checksum_paths([path, AUDIT_REPORT_PATH]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return events


def load_audit_log(path: Path = AUDIT_LOG_PATH) -> list[AuditEvent]:
    """Load JSONL audit events."""
    if not path.is_file():
        return []
    return [
        AuditEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_report(events: list[AuditEvent]) -> str:
    lines = [
        "# Audit Evidence Report",
        "",
        RESEARCH_DISCLAIMER,
        "",
        f"Events recorded: `{len(events)}`",
        "",
        "Audit fields exclude patient-identifiable data and raw arrays.",
        "",
    ]
    return "\n".join(lines)
