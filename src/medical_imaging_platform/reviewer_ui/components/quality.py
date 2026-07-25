"""Quality finding rendering helpers."""

from __future__ import annotations

from typing import Any


def render_quality(st_module: Any, findings: list[dict[str, Any]] | list[Any]) -> None:
    st_module.subheader("Quality Findings")
    if findings:
        st_module.json(findings)
    else:
        st_module.success("No quality findings returned by the governed API.")
