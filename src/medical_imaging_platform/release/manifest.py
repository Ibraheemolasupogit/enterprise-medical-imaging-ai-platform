"""Release manifest construction."""

from __future__ import annotations

import importlib.metadata
import json
import platform

# Git and Docker inspection commands use fixed argument lists and bounded timeouts.
import subprocess  # nosec B404
from datetime import UTC, datetime
from pathlib import Path

from medical_imaging_platform.release.checksums import checksum_paths
from medical_imaging_platform.release.dependencies import dependency_strategy
from medical_imaging_platform.release.models import (
    ContainerReleaseConfig,
    ReleaseCheckResult,
    ReleaseManifest,
    SmokeTestResult,
    ToolResult,
)
from medical_imaging_platform.release.status import aggregate_release_status


def deterministic_release_id(config: ContainerReleaseConfig, git_revision: str) -> str:
    """Build a deterministic release identifier."""
    return f"{config.policy_version}-{git_revision[:12] if git_revision else 'unknown'}"


def git_revision() -> str:
    return _git(["git", "rev-parse", "HEAD"]) or "unknown"


def git_dirty() -> bool:
    return bool(_git(["git", "status", "--porcelain"]))


def dependency_versions() -> dict[str, str]:
    """Record installed runtime dependency versions."""
    names = [
        "enterprise-medical-imaging-ai-platform",
        "fastapi",
        "monai",
        "numpy",
        "pydantic",
        "pydicom",
        "PyYAML",
        "scikit-learn",
        "SimpleITK",
        "streamlit",
        "torch",
        "uvicorn",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_release_manifest(
    config: ContainerReleaseConfig,
    test_results: list[ReleaseCheckResult],
    scan_results: list[ToolResult],
    smoke_results: SmokeTestResult,
) -> ReleaseManifest:
    revision = git_revision()
    image_inventory = {
        "api": _image_metadata(f"{config.api_image_name}:{config.image_tag}"),
        "reviewer_ui": _image_metadata(f"{config.reviewer_ui_image_name}:{config.image_tag}"),
    }
    return ReleaseManifest(
        release_id=deterministic_release_id(config, revision),
        git_revision=revision,
        git_dirty=git_dirty(),
        build_timestamp=datetime.now(UTC).replace(microsecond=0).isoformat(),
        python_version=platform.python_version(),
        release_status=aggregate_release_status(test_results, scan_results, smoke_results),
        dependency_strategy=dependency_strategy(config),
        dependency_versions=dependency_versions(),
        images=image_inventory,
        dockerfile_checksums=checksum_paths(
            [
                Path("docker/api/Dockerfile"),
                Path("docker/reviewer-ui/Dockerfile"),
                config.container_runtime_requirements,
            ]
        ),
        configuration_checksums=checksum_paths(
            [
                Path("config/container.yaml"),
                Path("config/container/api.yaml"),
                Path("config/container/reviewer_ui.yaml"),
            ]
        ),
        test_results=test_results,
        scan_results=scan_results,
        smoke_test_results=smoke_results,
    )


def _git(command: list[str]) -> str:
    # Fixed internal git command lists.
    completed = subprocess.run(  # nosec B603
        command, capture_output=True, text=True, timeout=15, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _image_metadata(image_ref: str) -> dict[str, str | int | None]:
    inspect = _docker_inspect(image_ref)
    name, _, tag = image_ref.partition(":")
    return {
        "name": name,
        "tag": tag or None,
        "reference": image_ref,
        "digest": inspect.get("Id"),
        "size_bytes": _parse_int(inspect.get("Size")),
        "virtual_size_bytes": _parse_int(inspect.get("VirtualSize")),
    }


def _docker_inspect(image_ref: str) -> dict[str, str]:
    # Fixed Docker image inspect command.
    completed = subprocess.run(  # nosec B603 B607
        ["docker", "image", "inspect", image_ref, "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return {}
    parsed = json.loads(completed.stdout)
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(value) for key, value in parsed.items()}


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
