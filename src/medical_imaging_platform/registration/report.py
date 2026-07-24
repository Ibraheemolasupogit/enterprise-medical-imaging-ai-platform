"""Registration Markdown report rendering."""

from __future__ import annotations

import json

from medical_imaging_platform.registration.models import RegistrationResult


def render_registration_report(result: RegistrationResult) -> str:
    """Render a deterministic engineering registration report."""
    lines = [
        "# Registration Report",
        "",
        f"- Registration run ID: `{result.registration_run_id}`",
        f"- Status: `{result.status}`",
        f"- Mode: `{result.mode}`",
        f"- Direction: `{result.registration_direction}`",
        f"- Fixed run ID: `{result.fixed.run_id}`",
        f"- Moving run ID: `{result.moving.run_id}`",
        f"- Initialisation: `{result.initialisation_method}`",
        f"- Processing duration seconds: `{result.processing_duration_seconds}`",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(
            {
                "before": result.metrics_before.model_dump(mode="json"),
                "after": result.metrics_after.model_dump(mode="json"),
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Transform",
        "",
        "```json",
        json.dumps(result.transform.model_dump(mode="json"), indent=2, sort_keys=True),
        "```",
        "",
        "## Findings",
        "",
    ]
    if result.findings:
        for finding in result.findings:
            lines.append(f"- `{finding.severity}` `{finding.rule_id}`: {finding.message}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "This is technical registration for research engineering evaluation only. Optimiser "
            "convergence, metric improvement, and synthetic translation recovery do not prove "
            "anatomical correctness or clinical suitability. No deformable, anatomy-constrained, "
            "radiologist-validated, diagnostic, or medical-device registration is implemented.",
            "",
            "## Recommended Next Action",
            "",
            result.recommended_next_action,
            "",
        ]
    )
    return "\n".join(lines)
