"""Provenance rendering helpers."""

from __future__ import annotations

from typing import Any


def render_provenance(st_module: Any, provenance: dict[str, Any]) -> None:
    st_module.subheader("Provenance")
    st_module.json(provenance or {})
