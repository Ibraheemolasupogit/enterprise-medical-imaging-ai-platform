"""Container dependency policy checks."""

from __future__ import annotations

from medical_imaging_platform.release.models import ContainerReleaseConfig, ReleaseCheckResult

PROHIBITED_CONTAINER_PACKAGES = (
    "nvidia-",
    "nvidia_",
    "cuda",
    "cudnn",
    "cublas",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nccl",
    "triton",
)
MINIMUM_SAFE_MONAI_VERSION = (1, 6, 0)
MINIMUM_SAFE_MONAI_VERSION_TEXT = "1.6.0"


def inspect_container_dependency_policy(
    config: ContainerReleaseConfig,
) -> list[ReleaseCheckResult]:
    """Validate container-only dependencies do not request CUDA/NVIDIA packages."""
    path = config.container_runtime_requirements
    if not path.exists():
        return [
            ReleaseCheckResult(
                check_id="CONTAINER-DEPS-REQUIREMENTS-EXISTS",
                status="FAIL",
                message=f"Container runtime requirements file is missing: {path}",
            )
        ]
    text = path.read_text(encoding="utf-8")
    normalized = _normalise_requirement_text(text)
    prohibited = [package for package in PROHIBITED_CONTAINER_PACKAGES if package in normalized]
    monai_version = _pinned_version(normalized, "monai")
    monai_safe = (
        monai_version is not None and _version_tuple(monai_version) >= MINIMUM_SAFE_MONAI_VERSION
    )
    return [
        ReleaseCheckResult(
            check_id="CONTAINER-DEPS-REQUIREMENTS-EXISTS",
            status="PASS",
            message="Container runtime requirements file is present.",
            details={"path": path.as_posix()},
        ),
        ReleaseCheckResult(
            check_id="CONTAINER-DEPS-PYTORCH-CPU-INDEX",
            status="PASS" if config.pytorch_cpu_index_url in text else "FAIL",
            message="Container dependencies use the official PyTorch CPU wheel index.",
            details={"index_url": config.pytorch_cpu_index_url},
        ),
        ReleaseCheckResult(
            check_id="CONTAINER-DEPS-PYTORCH-CPU-PIN",
            status="PASS" if f"torch=={config.pytorch_version}" in normalized else "FAIL",
            message="Container dependencies pin the configured CPU-only PyTorch wheel.",
            details={"pytorch_version": config.pytorch_version},
        ),
        ReleaseCheckResult(
            check_id="CONTAINER-DEPS-NO-CUDA-NVIDIA",
            status="PASS" if not prohibited else "FAIL",
            message="Container dependency inputs do not request CUDA or NVIDIA packages.",
            details={"prohibited_terms": prohibited},
        ),
        ReleaseCheckResult(
            check_id="CONTAINER-DEPS-MONAI-PATCHED-PIN",
            status="PASS" if monai_safe else "FAIL",
            message="Container dependencies pin MONAI at or above the patched minimum.",
            details={
                "minimum_safe_version": MINIMUM_SAFE_MONAI_VERSION_TEXT,
                "pinned_version": monai_version,
            },
        ),
    ]


def dependency_strategy(config: ContainerReleaseConfig) -> dict[str, str]:
    """Describe the container-only dependency strategy for evidence."""
    return {
        "requirements_file": config.container_runtime_requirements.as_posix(),
        "pytorch_version": config.pytorch_version,
        "pytorch_index_url": config.pytorch_cpu_index_url,
        "monai_version": _pinned_version(
            _normalise_requirement_text(
                config.container_runtime_requirements.read_text(encoding="utf-8")
            ),
            "monai",
        )
        or "not-pinned",
        "minimum_safe_monai_version": MINIMUM_SAFE_MONAI_VERSION_TEXT,
        "developer_dependency_workflow": (
            "pyproject.toml remains authoritative for local macOS development; containers "
            "install a CPU-only runtime wheelhouse then install the project wheel with --no-deps."
        ),
        "cuda_nvidia_policy": (
            "Container dependency inputs must not request CUDA, NVIDIA, cuDNN, NCCL, "
            "Triton or related GPU packages."
        ),
    }


def _normalise_requirement_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.split("#", maxsplit=1)[0].strip()
        if stripped:
            lines.append(stripped.lower())
    return "\n".join(lines)


def _pinned_version(normalized_requirements: str, package: str) -> str | None:
    prefix = f"{package.lower()}=="
    for line in normalized_requirements.splitlines():
        if line.startswith(prefix):
            version = line.removeprefix(prefix).split(";", maxsplit=1)[0].strip()
            return version or None
    return None


def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = version.split("+", maxsplit=1)[0].split(".")
    parsed: list[int] = []
    for part in parts[:3]:
        if not part.isdigit():
            return (0, 0, 0)
        parsed.append(int(part))
    while len(parsed) < 3:
        parsed.append(0)
    return (parsed[0], parsed[1], parsed[2])
