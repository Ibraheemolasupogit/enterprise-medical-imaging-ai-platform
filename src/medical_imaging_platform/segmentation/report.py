"""Markdown report rendering for segmentation experiments."""

from __future__ import annotations

from typing import Any


def render_segmentation_report(payload: dict[str, Any]) -> str:
    """Render deterministic research-only experiment evidence."""
    lines = [
        "# Segmentation Experiment Report",
        "",
        "Synthetic lesion segmentation engineering baseline only.",
        "",
        f"- Experiment ID: `{payload['experiment_id']}`",
        f"- Dataset ID: `{payload['dataset_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Device: `{payload['device']}`",
        f"- Seed: `{payload['random_seed']}`",
        f"- Model parameters: `{payload['model_parameter_count']}`",
        f"- Best epoch: `{payload['best_epoch']}`",
        f"- Best validation Dice: `{payload['best_validation_dice']}`",
        "",
        "## Quality Findings",
    ]
    for finding in payload["quality_findings"]:
        lines.append(
            f"- `{finding['severity']}` `{finding['rule_id']}` "
            f"`{finding['status']}`: {finding['message']}"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "- Trained only on synthetic engineering lesion masks.",
            "- Synthetic Dice does not demonstrate clinical segmentation performance.",
            "- No classification, diagnostic recommendation, or medical-device claim is made.",
            "",
            "## Recommended Action",
            str(payload["recommended_next_action"]),
            "",
        ]
    )
    return "\n".join(lines)
