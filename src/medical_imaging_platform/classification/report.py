"""Markdown reporting for classification experiments."""

from __future__ import annotations

from typing import Any


def render_classification_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Classification Experiment Report",
        "",
        "Synthetic binary lesion-presence classification engineering baseline only.",
        "",
        f"- Experiment ID: `{payload['experiment_id']}`",
        f"- Dataset ID: `{payload['dataset_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Device: `{payload['device']}`",
        f"- Seed: `{payload['random_seed']}`",
        f"- Model parameters: `{payload['model_parameter_count']}`",
        f"- Best epoch: `{payload['best_epoch']}`",
        f"- Calibration: `{payload['calibration']['method']}`",
        f"- Threshold: `{payload['threshold_policy']['selected_threshold']}`",
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
            "- Trained only on synthetic engineering lesion-presence labels.",
            "- This is not benign-versus-malignant classification.",
            "- Synthetic metrics do not demonstrate clinical performance.",
            "",
            "## Recommended Action",
            str(payload["recommended_next_action"]),
            "",
        ]
    )
    return "\n".join(lines)
