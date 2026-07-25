"""Local review-session export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_imaging_platform.reviewer_ui.models import ReviewerDecision, ReviewExportResult
from medical_imaging_platform.reviewer_ui.security import safe_review_output_dir
from medical_imaging_platform.synthetic.manifest import sha256_file


def export_review_session(
    *,
    decision: ReviewerDecision,
    evidence_summary: dict[str, Any],
    output_root: Path,
    overwrite: bool = False,
) -> ReviewExportResult:
    output_dir = safe_review_output_dir(output_root, decision.review_id)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Review export already exists: {output_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_path = output_dir / "review_decision.json"
    summary_path = output_dir / "reviewed_evidence_summary.json"
    _write_json(decision_path, decision.model_dump(mode="json"))
    _write_json(summary_path, _bounded_evidence_summary(evidence_summary))
    checksums = {
        "review_decision": sha256_file(decision_path),
        "reviewed_evidence_summary": sha256_file(summary_path),
    }
    report_path = output_dir / "review_report.md"
    _write_report(report_path, decision, checksums)
    checksums["review_report"] = sha256_file(report_path)
    return ReviewExportResult(
        output_directory=output_dir.name,
        checksums=checksums,
        files={
            "review_decision": decision_path.name,
            "reviewed_evidence_summary": summary_path.name,
            "review_report": report_path.name,
        },
    )


def _bounded_evidence_summary(payload: dict[str, Any]) -> dict[str, Any]:
    blocked = {"raw_arrays", "model_weights", "checkpoint_binary", "values"}
    return {key: value for key, value in payload.items() if key not in blocked}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_report(path: Path, decision: ReviewerDecision, checksums: dict[str, str]) -> None:
    lines = [
        "# Reviewer Session Report",
        "",
        "This is a local research engineering review artefact, not clinical approval.",
        "",
        f"- Review ID: `{decision.review_id}`",
        f"- Evidence type: `{decision.evidence_type}`",
        f"- Evidence ID: `{decision.evidence_id}`",
        f"- Model engineering label: `{decision.model_engineering_label}`",
        f"- Quality status: `{decision.quality_status}`",
        f"- Reviewer decision: `{decision.reviewer_decision}`",
        f"- Review timestamp: `{decision.review_timestamp}`",
        "",
        "## Checksums",
        "",
    ]
    lines.extend(f"- `{name}`: `{checksum}`" for name, checksum in sorted(checksums.items()))
    lines.append("")
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(path)
