"""Health and readiness routes."""
# mypy: disable-error-code="misc"

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from medical_imaging_platform import __version__
from medical_imaging_platform.api.dependencies import get_api_config
from medical_imaging_platform.api.errors import APIError
from medical_imaging_platform.api.models import DISCLAIMER, APIConfig
from medical_imaging_platform.synthetic.manifest import sha256_file

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    return {"status": "healthy", "request_id": request.state.request_id, "disclaimer": DISCLAIMER}


@router.get("/version")
def version() -> dict[str, str]:
    return {"package_version": __version__, "api_version": "0.1.0"}


@router.get("/ready")
def ready(config: Annotated[APIConfig, Depends(get_api_config)]) -> dict[str, object]:
    findings = readiness_findings(config)
    if any(not item["passed"] for item in findings):
        raise APIError(
            "API-NOTREADY-503",
            "API readiness checks failed.",
            status_code=503,
            details={
                str(item["rule_id"]): str(item["message"])
                for item in findings
                if not item["passed"]
            },
        )
    return {"status": "ready", "quality_findings": findings, "disclaimer": DISCLAIMER}


def readiness_findings(config: APIConfig) -> list[dict[str, object]]:
    findings = [
        _finding("API-QC-CONFIG-001", True, "Configuration loaded."),
        _finding(
            "API-QC-PATH-001", config.output_directory.parent.exists(), "Output parent exists."
        ),
    ]
    for name, path in {
        "segmentation_checkpoint": config.segmentation_checkpoint,
        "classification_checkpoint": config.classification_checkpoint,
        "classification_calibration": config.classification_calibration,
        "classification_threshold_policy": config.classification_threshold_policy,
        "longitudinal_config": config.longitudinal_config,
    }.items():
        if path is not None:
            exists = Path(path).exists()
            checksum = sha256_file(path) if exists and Path(path).is_file() else None
            findings.append(_finding("API-QC-MODEL-001", exists, f"{name} exists.", checksum))
    return findings


def _finding(
    rule_id: str, passed: bool, message: str, checksum: str | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {"rule_id": rule_id, "passed": passed, "message": message}
    if checksum is not None:
        payload["checksum"] = checksum
    return payload
