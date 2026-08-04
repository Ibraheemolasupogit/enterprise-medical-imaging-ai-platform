"""Load typed container release-assurance configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from medical_imaging_platform.release.models import ContainerReleaseConfig
from medical_imaging_platform.utils.config import ConfigError, load_yaml_file


def load_container_release_config(path: Path) -> ContainerReleaseConfig:
    """Load Milestone 13 container settings."""
    data = load_yaml_file(path)
    settings = data.get("settings")
    if not isinstance(settings, dict) or "container" not in settings:
        raise ConfigError(f"Missing settings.container in {path}")
    try:
        return ContainerReleaseConfig.model_validate(settings["container"])
    except ValidationError as exc:
        raise ConfigError(f"Invalid container configuration in {path}: {exc}") from exc
