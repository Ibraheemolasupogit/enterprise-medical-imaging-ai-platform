"""Markdown reports for localisation."""

from __future__ import annotations

import json

from medical_imaging_platform.localisation.models import LocalisationResult


def render_localisation_report(result: LocalisationResult) -> str:
    """Render a deterministic Markdown localisation report."""
    lines = [
        "# Localisation Report",
        "",
        f"- Run ID: `{result.localisation_run_id}`",
        f"- Status: `{result.overall_status}`",
        f"- Source run ID: `{result.source.source_run_id}`",
        f"- Source type: `{result.source.source_type}`",
        f"- Mode: `{result.localisation_mode}`",
        "",
        "## Predictions",
        "",
        "```json",
        json.dumps(
            {
                "left": result.left.model_dump(mode="json"),
                "right": result.right.model_dump(mode="json"),
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(
            {side: value.model_dump(mode="json") for side, value in result.metrics.items()},
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Findings",
        "",
    ]
    if result.quality_findings:
        for finding in result.quality_findings:
            lines.append(
                f"- `{finding.severity}` `{finding.rule_id}` `{finding.status}`: {finding.message}"
            )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "This is an atlas/geometry baseline for engineering adrenal-region placeholders. "
            "It is not a clinically validated adrenal localiser, organ segmentation, lesion "
            "detector, diagnostic system, or medical-device workflow.",
            "",
            "## Recommended Next Action",
            "",
            result.recommended_next_action,
            "",
        ]
    )
    return "\n".join(lines)
