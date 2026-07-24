"""Markdown report rendering for longitudinal analysis."""

from __future__ import annotations

from typing import Any


def render_longitudinal_report(payload: dict[str, Any]) -> str:
    """Render a concise research-only longitudinal report."""
    summary = payload["summary"]
    labels = ", ".join(summary["engineering_labels"])
    disclaimer = (
        "This platform is a research and engineering demonstrator. Outputs are intended for "
        "technical evaluation and human review only and must not be used for clinical diagnosis "
        "or patient-management decisions."
    )
    return f"""# Longitudinal Analysis Report

{disclaimer}

## Summary

- Analysis ID: `{payload["analysis_id"]}`
- Case ID: `{payload["pair_manifest"]["case_id"]}`
- Side: `{payload["pair_manifest"]["side"]}`
- Status: `{payload["status"]}`
- Engineering labels: `{labels}`

## Boundaries

Labels are engineering categories only: new, increased, stable, reduced, resolved, or indeterminate.
They are not RECIST, disease progression, treatment response, diagnosis, or clinical decision
support.

## Recommended Next Action

{payload["recommended_next_action"]}
"""
