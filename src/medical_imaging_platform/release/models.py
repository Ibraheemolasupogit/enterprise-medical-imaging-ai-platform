"""Typed models for local release assurance."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Status = Literal["PASS", "FAIL", "WARN", "INCOMPLETE", "UNAVAILABLE", "ERROR", "SKIPPED"]

DISCLAIMER = (
    "This is local engineering release evidence for a research demonstrator only. It is not a "
    "diagnostic system, approved medical device, clinical deployment, or production release."
)


class ContainerReleaseConfig(BaseModel):
    """Typed Milestone 13 container and release-assurance configuration."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = Field(min_length=1)
    api_image_name: str = Field(pattern=r"^[a-z0-9][a-z0-9._/-]*$")
    reviewer_ui_image_name: str = Field(pattern=r"^[a-z0-9][a-z0-9._/-]*$")
    image_tag: str = Field(min_length=1)
    container_runtime_requirements: Path
    pytorch_cpu_index_url: str
    pytorch_version: str
    api_port: int = Field(gt=0, le=65535)
    reviewer_ui_port: int = Field(gt=0, le=65535)
    api_health_path: str
    api_readiness_path: str
    api_version_path: str = "/version"
    api_openapi_path: str = "/openapi.json"
    reviewer_health_path: str
    read_only_root_filesystem: bool
    run_as_non_root: bool
    required_uid: int = Field(gt=0)
    required_gid: int = Field(gt=0)
    drop_all_capabilities: bool
    no_new_privileges: bool
    maximum_memory: str = Field(min_length=1)
    maximum_cpus: float = Field(gt=0)
    output_mount: PurePosixPath
    evidence_mount: PurePosixPath
    checkpoint_mount: PurePosixPath
    temporary_directory: PurePosixPath
    release_output_directory: Path = Path("reports/generated/releases")
    compose_project_name: str = Field(min_length=1)
    sbom_format: Literal["cyclonedx-json", "spdx-json"]
    vulnerability_fail_severities: list[str] = Field(min_length=1)
    scanner_timeout_seconds: int = Field(gt=0)
    smoke_timeout_seconds: int = Field(gt=0)
    allow_docker_unavailable: bool = True

    @field_validator(
        "api_health_path",
        "api_readiness_path",
        "api_version_path",
        "api_openapi_path",
        "reviewer_health_path",
    )
    @classmethod
    def health_path_local(cls, value: str) -> str:
        if not value.startswith("/") or "://" in value or "*" in value:
            raise ValueError("Health paths must be local absolute HTTP paths.")
        return value

    @field_validator("container_runtime_requirements")
    @classmethod
    def requirements_path_local(cls, value: Path) -> Path:
        text = value.as_posix()
        if value.is_absolute() or text.startswith("..") or "*" in text:
            raise ValueError("Container requirements path must be a local repository path.")
        return value

    @field_validator("pytorch_cpu_index_url")
    @classmethod
    def pytorch_index_cpu(cls, value: str) -> str:
        if value != "https://download.pytorch.org/whl/cpu":
            raise ValueError("Container PyTorch index must use the official CPU wheel index.")
        return value

    @field_validator("pytorch_version")
    @classmethod
    def pytorch_version_cpu(cls, value: str) -> str:
        if "+cpu" not in value:
            raise ValueError("Container PyTorch version must pin a CPU wheel.")
        return value

    @field_validator("output_mount", "evidence_mount", "checkpoint_mount", "temporary_directory")
    @classmethod
    def mount_absolute_and_specific(cls, value: PurePosixPath) -> PurePosixPath:
        text = value.as_posix()
        if not text.startswith("/") or text in {"/", "/*"} or "*" in text:
            raise ValueError("Container mount destinations must be specific absolute paths.")
        return value

    @field_validator("vulnerability_fail_severities")
    @classmethod
    def severities_known(cls, value: list[str]) -> list[str]:
        allowed = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        upper = [item.upper() for item in value]
        if any(item not in allowed for item in upper):
            raise ValueError("Unsupported vulnerability severity.")
        return upper

    @model_validator(mode="after")
    def relationships_valid(self) -> ContainerReleaseConfig:
        if self.api_port == self.reviewer_ui_port:
            raise ValueError("API and reviewer UI ports must be distinct.")
        if self.required_uid == 0 or self.required_gid == 0:
            raise ValueError("Container UID/GID must be non-root.")
        if not self.read_only_root_filesystem:
            raise ValueError("Read-only root filesystem must be enabled.")
        if not self.run_as_non_root:
            raise ValueError("Containers must run as non-root.")
        if not self.drop_all_capabilities:
            raise ValueError("All Linux capabilities must be dropped.")
        if not self.no_new_privileges:
            raise ValueError("no-new-privileges must be enabled.")
        return self


class ReleaseCheckResult(BaseModel):
    """One machine-readable release-assurance check result."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    status: Status
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result from an optional external release scanner."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    available: bool
    status: Status
    version: str | None = None
    command: list[str] = Field(default_factory=list)
    output: str = ""
    findings: list[dict[str, Any]] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class SmokeTestResult(BaseModel):
    """Container smoke-test evidence."""

    model_config = ConfigDict(extra="forbid")

    status: Status
    executed: bool
    steps: list[ReleaseCheckResult]
    logs_redacted: bool = True


class ReleaseManifest(BaseModel):
    """Deterministic release manifest payload."""

    model_config = ConfigDict(extra="forbid")

    release_id: str
    git_revision: str
    git_dirty: bool
    build_timestamp: str
    python_version: str
    release_status: Status
    dependency_strategy: dict[str, Any]
    dependency_versions: dict[str, str]
    images: dict[str, dict[str, Any]]
    dockerfile_checksums: dict[str, str]
    configuration_checksums: dict[str, str]
    test_results: list[ReleaseCheckResult]
    scan_results: list[ToolResult]
    smoke_test_results: SmokeTestResult
    disclaimer: str = DISCLAIMER
