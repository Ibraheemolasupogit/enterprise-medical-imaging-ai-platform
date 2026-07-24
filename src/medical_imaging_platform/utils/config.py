"""Typed YAML configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from medical_imaging_platform.ingestion.models import DicomIngestionConfig
from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class ConfigError(MedicalImagingPlatformError):
    """Raised when repository configuration cannot be loaded or validated."""


class PlatformConfig(BaseModel):
    """Shared schema for Milestone 1 placeholder configuration files."""

    model_config = ConfigDict(extra="forbid")

    config_name: str = Field(min_length=1)
    milestone: int = Field(ge=1)
    status: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    settings: dict[str, Any]
    safeguards: list[str] = Field(min_length=1)
    future_capabilities: list[str] = Field(min_length=1)

    @field_validator("status")
    @classmethod
    def status_must_be_planned_or_foundation(cls, value: str) -> str:
        allowed = {"foundation", "planned_not_implemented"}
        if value not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return value

    @field_validator("settings")
    @classmethod
    def settings_must_not_be_empty(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("settings must document at least one placeholder setting")
        return value


class RepositoryConfigSet(BaseModel):
    """Validated collection of repository configuration files."""

    configs: dict[str, PlatformConfig]
    required_files: ClassVar[tuple[str, ...]] = (
        "platform.yaml",
        "data.yaml",
        "preprocessing.yaml",
        "registration.yaml",
        "localisation.yaml",
        "segmentation.yaml",
        "classification.yaml",
        "evaluation.yaml",
        "monitoring.yaml",
        "governance.yaml",
    )


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load one YAML file and return a mapping."""
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Configuration path is not a file: {path}")

    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ConfigError(f"Configuration file must contain a YAML mapping: {path}")
    return parsed


def load_config(path: Path) -> PlatformConfig:
    """Load and validate one Milestone 1 configuration file."""
    data = load_yaml_file(path)
    try:
        return PlatformConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {path}: {exc}") from exc


def validate_repository_configs(config_dir: Path) -> RepositoryConfigSet:
    """Validate every required repository configuration file."""
    if not config_dir.exists():
        raise ConfigError(f"Configuration directory not found: {config_dir}")
    if not config_dir.is_dir():
        raise ConfigError(f"Configuration path is not a directory: {config_dir}")

    configs: dict[str, PlatformConfig] = {}
    for filename in RepositoryConfigSet.required_files:
        configs[filename] = load_config(config_dir / filename)

    return RepositoryConfigSet(configs=configs)


def load_dicom_ingestion_config(path: Path) -> DicomIngestionConfig:
    """Load typed Milestone 3 DICOM ingestion settings from data.yaml."""
    data = load_yaml_file(path)
    settings = data.get("settings")
    if not isinstance(settings, dict) or "dicom_ingestion" not in settings:
        raise ConfigError(f"Missing settings.dicom_ingestion in {path}")
    try:
        return DicomIngestionConfig.model_validate(settings["dicom_ingestion"])
    except ValidationError as exc:
        raise ConfigError(f"Invalid DICOM ingestion configuration in {path}: {exc}") from exc
