"""Read-only review evidence services."""

from __future__ import annotations

from medical_imaging_platform.api.errors import APIError
from medical_imaging_platform.api.models import APIConfig
from medical_imaging_platform.classification.export import validate_classification_experiment
from medical_imaging_platform.longitudinal.export import validate_longitudinal_analysis
from medical_imaging_platform.segmentation.export import validate_segmentation_experiment


def review_segmentation(experiment_id: str, config: APIConfig) -> dict[str, object]:
    return _review(experiment_id, config, "segmentation", validate_segmentation_experiment)


def review_classification(experiment_id: str, config: APIConfig) -> dict[str, object]:
    return _review(experiment_id, config, "classification", validate_classification_experiment)


def review_longitudinal(analysis_id: str, config: APIConfig) -> dict[str, object]:
    return _review(analysis_id, config, "longitudinal", validate_longitudinal_analysis)


def _review(
    evidence_id: str, config: APIConfig, evidence_type: str, validator: object
) -> dict[str, object]:
    if "/" in evidence_id or "\\" in evidence_id or ".." in evidence_id:
        raise APIError("API-AUTH-403", "Invalid evidence identifier.", status_code=403)
    candidates = [
        root.resolve() / evidence_type / evidence_id for root in config.allowed_evidence_roots
    ]
    candidates += [root.resolve() / evidence_id for root in config.allowed_evidence_roots]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            try:
                manifest = validator(candidate)  # type: ignore[operator]
            except Exception as exc:
                raise APIError(
                    "API-INTEGRITY-409", "Evidence integrity validation failed.", status_code=409
                ) from exc
            return _summary(evidence_type, manifest)
    raise APIError("API-NOTFOUND-404", "Evidence was not found.", status_code=404)


def _summary(evidence_type: str, manifest: dict[str, object]) -> dict[str, object]:
    if evidence_type == "longitudinal":
        return {
            "evidence_type": evidence_type,
            "status": manifest["status"],
            "summary": manifest["summary"],
            "quality_findings": manifest["quality_findings"],
            "provenance": _longitudinal_public_provenance(manifest["pair_manifest"]),
        }
    return {
        "evidence_type": evidence_type,
        "status": manifest["status"],
        "summary": {
            "experiment_id": manifest["experiment_id"],
            "dataset_id": manifest["dataset_id"],
        },
        "quality_findings": manifest.get("quality_findings", []),
        "provenance": {
            "configuration_checksum": manifest.get("configuration_checksum"),
            "dataset_manifest_checksum": manifest.get("dataset_manifest_checksum"),
        },
    }


def _longitudinal_public_provenance(pair_manifest: object) -> dict[str, object]:
    if not isinstance(pair_manifest, dict):
        return {}
    allowed = {
        "analysis_id",
        "case_id",
        "research_subject_id",
        "side",
        "previous_timepoint",
        "current_timepoint",
        "registration_run_id",
        "upstream_quality_statuses",
        "source_checksums",
        "policy_version",
    }
    return {key: value for key, value in pair_manifest.items() if key in allowed}
