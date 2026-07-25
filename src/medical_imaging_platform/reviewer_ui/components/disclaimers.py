"""Reusable reviewer UI disclaimer component."""

from __future__ import annotations

from typing import Any

from medical_imaging_platform.api.models import DISCLAIMER


def render_disclaimer(st_module: Any) -> None:
    st_module.warning(DISCLAIMER)
