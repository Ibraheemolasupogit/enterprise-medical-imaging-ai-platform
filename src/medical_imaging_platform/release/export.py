"""Export local release evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from medical_imaging_platform.release.checksums import checksum_paths
from medical_imaging_platform.release.models import (
    ContainerReleaseConfig,
    ReleaseCheckResult,
    ReleaseManifest,
    Status,
)
from medical_imaging_platform.release.report import release_report_markdown


def export_release_evidence(
    config: ContainerReleaseConfig,
    manifest: ReleaseManifest,
    *,
    overwrite: bool = False,
) -> Path:
    """Write deterministic release evidence files under ignored generated outputs."""
    output_dir = config.release_output_directory / manifest.release_id
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Release evidence already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, Any] = {
        "release_manifest.json": manifest.model_dump(mode="json"),
        "image_inventory.json": manifest.images,
        "container_dependency_strategy.json": manifest.dependency_strategy,
        "dependency_inventory.json": manifest.dependency_versions,
        "container_security_checks.json": [
            result.model_dump(mode="json") for result in manifest.test_results
        ],
        "smoke_test_results.json": manifest.smoke_test_results.model_dump(mode="json"),
    }
    for filename, payload in payloads.items():
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (output_dir / "release_report.md").write_text(
        release_report_markdown(manifest), encoding="utf-8"
    )
    checksum_targets = [
        path for path in output_dir.iterdir() if path.name != "checksum_manifest.json"
    ]
    (output_dir / "checksum_manifest.json").write_text(
        json.dumps(checksum_paths(checksum_targets), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def validate_release_evidence(path: Path) -> list[ReleaseCheckResult]:
    """Validate required release evidence files and checksum presence."""
    required = [
        "release_manifest.json",
        "image_inventory.json",
        "container_dependency_strategy.json",
        "dependency_inventory.json",
        "container_security_checks.json",
        "smoke_test_results.json",
        "checksum_manifest.json",
        "release_report.md",
    ]
    checks = [
        ReleaseCheckResult(
            check_id=f"RELEASE-EVIDENCE-{filename.upper()}",
            status="PASS" if (path / filename).is_file() else "FAIL",
            message=f"{filename} is present.",
        )
        for filename in required
    ]
    if (path / "checksum_manifest.json").is_file():
        parsed = json.loads((path / "checksum_manifest.json").read_text(encoding="utf-8"))
        checks.append(
            ReleaseCheckResult(
                check_id="RELEASE-EVIDENCE-CHECKSUMS",
                status="PASS" if isinstance(parsed, dict) and len(parsed) >= 6 else "FAIL",
                message="Checksum manifest records generated evidence files.",
            )
        )
    if (path / "release_manifest.json").is_file():
        manifest = json.loads((path / "release_manifest.json").read_text(encoding="utf-8"))
        status = str(manifest.get("release_status", "ERROR"))
        evidence_status: Status = (
            cast(Status, status) if status in {"PASS", "FAIL", "INCOMPLETE", "ERROR"} else "ERROR"
        )
        checks.append(
            ReleaseCheckResult(
                check_id="RELEASE-EVIDENCE-OVERALL-STATUS",
                status=evidence_status,
                message=f"Release manifest overall status is {status}.",
            )
        )
    return checks
