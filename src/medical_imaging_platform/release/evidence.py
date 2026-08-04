"""Persist and load canonical local release evidence."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from medical_imaging_platform.release.models import (
    ContainerReleaseConfig,
    SmokeTestResult,
    ToolResult,
)

CANONICAL_TOOL_KEYS = (
    "gitleaks",
    "pip-audit",
    "syft-api",
    "syft-reviewer-ui",
    "trivy-api",
    "trivy-reviewer-ui",
    "hadolint",
)

MANDATORY_TOOL_KEYS = {
    "gitleaks",
    "pip-audit",
    "syft-api",
    "syft-reviewer-ui",
    "trivy-api",
    "trivy-reviewer-ui",
}


def latest_evidence_dir(config: ContainerReleaseConfig) -> Path:
    """Return the ignored directory used for latest local release evidence."""
    return config.release_output_directory / "latest-evidence"


def write_tool_evidence(config: ContainerReleaseConfig, key: str, result: ToolResult) -> Path:
    """Persist one canonical tool result."""
    if key not in CANONICAL_TOOL_KEYS:
        raise ValueError(f"Unsupported canonical tool evidence key: {key}")
    output_dir = latest_evidence_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump(mode="json")
    payload["details"] = {
        **payload.get("details", {}),
        "canonical_key": key,
        "mandatory": key in MANDATORY_TOOL_KEYS,
    }
    output_path = output_dir / f"{key}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_smoke_evidence(config: ContainerReleaseConfig, result: SmokeTestResult) -> Path:
    """Persist latest container smoke evidence."""
    output_dir = latest_evidence_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "container-smoke.json"
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_canonical_tool_evidence(config: ContainerReleaseConfig) -> list[ToolResult]:
    """Load exactly one result for each canonical scanner/SBOM/image-scan target."""
    output_dir = latest_evidence_dir(config)
    return [_load_tool(output_dir / f"{key}.json", key) for key in CANONICAL_TOOL_KEYS]


def load_smoke_evidence(config: ContainerReleaseConfig) -> SmokeTestResult:
    """Load persisted smoke evidence, or return an incomplete placeholder when absent."""
    output_path = latest_evidence_dir(config) / "container-smoke.json"
    if not output_path.is_file():
        return SmokeTestResult(status="INCOMPLETE", executed=False, steps=[])
    try:
        return SmokeTestResult.model_validate_json(output_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return SmokeTestResult(status="ERROR", executed=False, steps=[])


def _load_tool(path: Path, key: str) -> ToolResult:
    mandatory = key in MANDATORY_TOOL_KEYS
    if not path.is_file():
        tool, target = _tool_and_target(key)
        return ToolResult(
            tool=tool,
            available=False,
            status="INCOMPLETE" if mandatory else "UNAVAILABLE",
            output=f"Latest canonical evidence is missing for {key}.",
            details={"canonical_key": key, "target": target, "mandatory": mandatory},
        )
    try:
        parsed = ToolResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        tool, target = _tool_and_target(key)
        return ToolResult(
            tool=tool,
            available=False,
            status="ERROR",
            output=f"Latest canonical evidence could not be parsed for {key}.",
            details={"canonical_key": key, "target": target, "mandatory": mandatory},
        )
    return parsed.model_copy(
        update={
            "details": {
                **parsed.details,
                "canonical_key": key,
                "mandatory": mandatory,
            }
        }
    )


def _tool_and_target(key: str) -> tuple[str, str | None]:
    if "-" not in key:
        return key, None
    tool, _, target = key.partition("-")
    return tool, target
