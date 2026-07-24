"""Deterministic quality report writers."""

from __future__ import annotations

import json
from pathlib import Path

from medical_imaging_platform.quality_control.models import QualityReport


def write_quality_reports(
    report: QualityReport, output_dir: Path, *, overwrite: bool = False
) -> None:
    """Write deterministic JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "quality_report.json"
    markdown_path = output_dir / "quality_report.md"
    if not overwrite and (json_path.exists() or markdown_path.exists()):
        raise FileExistsError(f"Quality report output already exists in {output_dir}")
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: QualityReport) -> str:
    """Render a human-readable quality report."""
    lines = [
        "# DICOM Quality Report",
        "",
        f"- Report ID: `{report.quality_report_id}`",
        f"- Study UID: `{report.study_instance_uid}`",
        f"- Series UID: `{report.series_instance_uid}`",
        f"- Status: `{report.status}`",
        f"- Quality score: `{report.quality_score}`",
        f"- Evaluated files: `{report.evaluated_file_count}`",
        f"- Ordering strategy: `{report.ordering_strategy}`",
        "",
        "## Findings",
        "",
    ]
    for finding in sorted(report.findings, key=lambda item: (item.severity, item.rule_id)):
        lines.append(
            f"- `{finding.severity}` `{finding.rule_id}` `{finding.status}`: {finding.message}"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(report.metrics, indent=2, sort_keys=True),
            "```",
            "",
            "## Limitations",
            "",
            "This is an engineering data-quality report, not a clinical image-quality assessment. "
            "A passing score does not mean the scan is diagnostically adequate. Burned-in "
            "annotation metadata cannot guarantee absence of pixel PHI.",
            "",
        ]
    )
    return "\n".join(lines)
