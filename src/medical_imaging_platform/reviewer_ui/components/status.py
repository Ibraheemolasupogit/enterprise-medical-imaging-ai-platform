"""Status rendering helpers."""

from __future__ import annotations

from typing import Any


def render_status(st_module: Any, title: str, payload: dict[str, Any]) -> None:
    st_module.subheader(title)
    st_module.json(payload)
