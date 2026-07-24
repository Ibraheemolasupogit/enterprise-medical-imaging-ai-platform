"""Human-readable preprocessing reports."""

from __future__ import annotations

import json

from medical_imaging_platform.preprocessing.models import PreprocessingResult


def render_preprocessing_report(result: PreprocessingResult) -> str:
    """Render a deterministic Markdown preprocessing report."""
    lines = [
        "# CT Preprocessing Report",
        "",
        f"- Run ID: `{result.run_id}`",
        f"- Study UID: `{result.study_instance_uid}`",
        f"- Series UID: `{result.series_instance_uid}`",
        f"- Source QC report: `{result.source_quality_report_id}`",
        f"- Source QC status: `{result.source_quality_status}`",
        f"- Output shape [z, y, x]: `{result.volume_shape}`",
        f"- Spacing [z_mm, y_mm, x_mm]: `{result.spacing_mm}`",
        f"- Orientation classification: `{result.orientation_classification}`",
        f"- Override used: `{result.override_used}`",
        "",
        "## Processing Trace",
        "",
        "```json",
        json.dumps(
            {
                "pixel_conversion": result.pixel_conversion.model_dump(mode="json"),
                "intensity_transform": result.intensity_transform.model_dump(mode="json"),
                "crop_padding": result.crop_padding.model_dump(mode="json"),
                "geometry": result.geometry.model_dump(mode="json"),
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Checksums",
        "",
    ]
    for path, checksum in sorted(result.checksums.items()):
        lines.append(f"- `{path}`: `{checksum}`")
    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    if result.warnings:
        for warning in sorted(result.warnings):
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "This is a deterministic engineering preprocessing report for synthetic or "
            "de-identified research data. It is not a diagnostic result, does not perform "
            "anatomical localisation, registration, segmentation, NIfTI conversion, spatial "
            "resampling, or medical-device quality assurance.",
            "",
        ]
    )
    return "\n".join(lines)
