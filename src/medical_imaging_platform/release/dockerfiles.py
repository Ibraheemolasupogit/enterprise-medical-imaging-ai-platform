"""Static Dockerfile assurance checks."""

from __future__ import annotations

import re
from pathlib import Path

from medical_imaging_platform.release.models import ContainerReleaseConfig, ReleaseCheckResult


def inspect_dockerfile(
    path: Path, expected_title: str, config: ContainerReleaseConfig
) -> list[ReleaseCheckResult]:
    """Inspect one Dockerfile for local release controls."""
    if not path.exists():
        return [_fail("DOCKERFILE-EXISTS", f"Missing Dockerfile: {path}")]
    text = path.read_text(encoding="utf-8")
    checks = [
        _check(
            "DOCKERFILE-MULTISTAGE",
            " AS builder" in text and " AS runtime" in text,
            "Multi-stage build is present.",
        ),
        _check(
            "DOCKERFILE-NO-LATEST",
            ":latest" not in text and " latest" not in text,
            "No latest base image tag is used.",
        ),
        _check(
            "DOCKERFILE-PYTHON312",
            "python:3.12" in text,
            "Python 3.12-compatible base image is declared.",
        ),
        _check(
            "DOCKERFILE-NONROOT",
            f"USER {config.required_uid}:{config.required_gid}" in text,
            "Runtime user is non-root.",
        ),
        _check(
            "DOCKERFILE-WORKDIR", "WORKDIR /app" in text, "Fixed runtime working directory is set."
        ),
        _check(
            "DOCKERFILE-BYTECODE",
            "PYTHONDONTWRITEBYTECODE=1" in text,
            "Deterministic Python bytecode environment is set.",
        ),
        _check(
            "DOCKERFILE-UNBUFFERED",
            "PYTHONUNBUFFERED=1" in text,
            "Unbuffered Python output is set.",
        ),
        _check(
            "DOCKERFILE-EXEC-CMD",
            bool(re.search(r"^CMD \[", text, flags=re.MULTILINE)),
            "Runtime command uses exec form.",
        ),
        _check(
            "DOCKERFILE-LABELS",
            "org.opencontainers.image.title" in text and expected_title in text,
            "OCI labels are present.",
        ),
        _check(
            "DOCKERFILE-NO-SSH",
            "sshd" not in text.lower() and "openssh" not in text.lower(),
            "No SSH server is installed.",
        ),
        _check(
            "DOCKERFILE-NO-MODEL-WEIGHTS",
            not re.search(r"COPY .*\\.(pt|pth|onnx|h5)", text),
            "No model weights are copied.",
        ),
        _check(
            "DOCKERFILE-CONTAINER-REQUIREMENTS",
            "requirements/container-runtime.txt" in text,
            "Container-only runtime requirements are copied before project source.",
        ),
        _check(
            "DOCKERFILE-PROJECT-NO-DEPS",
            "--no-deps" in text,
            "Project wheel installs without re-resolving runtime dependencies.",
        ),
        _check(
            "DOCKERFILE-NO-DEV-DEPS",
            '".[dev]"' not in text and "'.[dev]'" not in text,
            "Development/test extras are not installed in runtime images.",
        ),
    ]
    return checks


def inspect_required_dockerfiles(config: ContainerReleaseConfig) -> list[ReleaseCheckResult]:
    """Inspect both Milestone 13 Dockerfiles."""
    return [
        *inspect_dockerfile(Path("docker/api/Dockerfile"), config.api_image_name, config),
        *inspect_dockerfile(
            Path("docker/reviewer-ui/Dockerfile"), config.reviewer_ui_image_name, config
        ),
    ]


def _check(check_id: str, passed: bool, message: str) -> ReleaseCheckResult:
    return ReleaseCheckResult(
        check_id=check_id, status="PASS" if passed else "FAIL", message=message
    )


def _fail(check_id: str, message: str) -> ReleaseCheckResult:
    return ReleaseCheckResult(check_id=check_id, status="FAIL", message=message)
