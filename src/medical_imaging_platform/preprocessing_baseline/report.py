"""Markdown reports for M19 preprocessing baseline evidence."""

from __future__ import annotations

import json

from medical_imaging_platform.preprocessing_baseline.models import PreprocessingBaselineResult


def render_preprocessing_baseline_report(result: PreprocessingBaselineResult) -> str:
    """Render a deterministic engineering-only M19 baseline report."""
    lines = [
        "# Medical Image Preprocessing Baseline Report",
        "",
        f"- Baseline run ID: `{result.baseline_run_id}`",
        f"- Policy version: `{result.policy_version}`",
        f"- Source count: `{result.source_count}`",
        "",
        "## Scope",
        "",
        "This M19 baseline operates on already-generated preprocessed NumPy volumes. It does not "
        "replace canonical DICOM preprocessing and does not alter source preprocessing artefacts.",
        "",
        "## MONAI Comparison",
        "",
        json.dumps(result.monai_comparison, indent=2, sort_keys=True),
        "",
        "## Source Evidence",
        "",
    ]
    for source in result.sources:
        lines.extend(
            [
                f"### {source.source_preprocessing_run_id}",
                "",
                f"- Source directory: `{source.source_output_dir}`",
                f"- Volume shape: `{source.volume_shape}`",
                f"- Spacing mm: `{source.spacing_mm}`",
                f"- Transform chain: `{[record.name for record in source.transform_chain]}`",
                "",
                "```json",
                json.dumps(
                    {
                        "before_metrics": source.before_metrics.model_dump(mode="json"),
                        "after_metrics": source.after_metrics.model_dump(mode="json"),
                        "metric_deltas": source.metric_deltas,
                        "warnings": source.warnings,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Checksums",
            "",
        ]
    )
    for key, checksum in sorted(result.checksums.items()):
        lines.append(f"- `{key}`: `{checksum}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in result.limitations:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)
