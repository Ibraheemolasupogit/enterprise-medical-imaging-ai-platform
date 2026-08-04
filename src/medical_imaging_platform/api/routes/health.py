"""Health and readiness routes."""
# mypy: disable-error-code="misc"

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import PlainTextResponse

from medical_imaging_platform import __version__
from medical_imaging_platform.api.dependencies import get_api_config
from medical_imaging_platform.api.errors import APIError
from medical_imaging_platform.api.models import DISCLAIMER, APIConfig
from medical_imaging_platform.api.observability import metric_labels_safe
from medical_imaging_platform.synthetic.manifest import sha256_file

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    return {
        "status": "alive",
        "liveness": "process_alive",
        "request_id": request.state.request_id,
        "startup_status": getattr(request.app.state, "startup_status", {}),
        "disclaimer": DISCLAIMER,
    }


@router.get("/version")
def version() -> dict[str, str]:
    return {"package_version": __version__, "api_version": "0.1.0"}


@router.get("/ready")
def ready(config: Annotated[APIConfig, Depends(get_api_config)]) -> dict[str, object]:
    findings = readiness_findings(config)
    degraded = any(item["severity"] == "WARN" for item in findings)
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
    return {
        "status": "degraded" if degraded else "ready",
        "readiness": "dependencies_available",
        "degraded": degraded,
        "quality_findings": findings,
        "disclaimer": DISCLAIMER,
    }


@router.get("/metrics", response_class=PlainTextResponse)
def metrics(
    request: Request,
    config: Annotated[APIConfig, Depends(get_api_config)],
    x_metrics_token: Annotated[str | None, Header(alias="X-Metrics-Token")] = None,
) -> PlainTextResponse:
    if not config.enable_metrics_endpoint:
        raise APIError("API-METRICS-404", "Metrics endpoint is disabled.", status_code=404)
    if config.metrics_access_token and x_metrics_token != config.metrics_access_token:
        raise APIError("API-METRICS-403", "Metrics endpoint token is invalid.", status_code=403)
    registry = request.app.state.metrics_registry
    findings = readiness_findings(config)
    registry.set_readiness(
        ready=not any(not item["passed"] for item in findings),
        degraded=any(item["severity"] == "WARN" for item in findings),
    )
    payload = registry.render_prometheus()
    if not metric_labels_safe(payload):
        raise APIError("API-METRICS-500", "Unsafe metric labels detected.", status_code=500)
    return PlainTextResponse(payload, media_type="text/plain; version=0.0.4")


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
    if not any(
        [
            config.segmentation_checkpoint,
            config.classification_checkpoint,
            config.classification_calibration,
            config.classification_threshold_policy,
        ]
    ):
        findings.append(
            _finding(
                "API-QC-DEGRADED-001",
                True,
                "No model artefacts configured; governed inference readiness is degraded.",
                severity="WARN",
            )
        )
    return findings


def _finding(
    rule_id: str,
    passed: bool,
    message: str,
    checksum: str | None = None,
    severity: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "rule_id": rule_id,
        "passed": passed,
        "message": message,
        "severity": severity or ("INFO" if passed else "ERROR"),
    }
    if checksum is not None:
        payload["checksum"] = checksum
    return payload
