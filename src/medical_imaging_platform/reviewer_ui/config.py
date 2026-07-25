"""Reviewer UI configuration loading."""

from __future__ import annotations

from pathlib import Path

from medical_imaging_platform.reviewer_ui.models import ReviewerUIConfig
from medical_imaging_platform.utils.config import load_reviewer_ui_config as _load


def load_reviewer_ui_config(path: Path) -> ReviewerUIConfig:
    """Load typed Milestone 12 reviewer UI settings."""
    return _load(path)
