"""Top-level longitudinal synthetic lesion analysis pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from medical_imaging_platform.longitudinal.change import calculate_changes
from medical_imaging_platform.longitudinal.export import (
    write_longitudinal_evidence,
    write_manifest,
)
from medical_imaging_platform.longitudinal.matching import match_lesions
from medical_imaging_platform.longitudinal.measurements import (
    measure_components,
    validate_binary_mask,
)
from medical_imaging_platform.longitudinal.models import LongitudinalConfig, PairManifest
from medical_imaging_platform.longitudinal.quality import (
    evaluate_quality,
    forced_indeterminate_reasons,
)
from medical_imaging_platform.longitudinal.visualisation import write_review_arrays
from medical_imaging_platform.synthetic.manifest import sha256_file, stable_timestamp


class LongitudinalAnalysisError(ValueError):
    """Raised when longitudinal analysis is invalid."""


def analyse_longitudinal_pair(
    *,
    previous_mask_path: Path,
    current_mask_path: Path,
    previous_spacing_mm: tuple[float, float, float],
    current_spacing_mm: tuple[float, float, float],
    case_id: str,
    research_subject_id: str,
    side: str,
    previous_timepoint: str,
    current_timepoint: str,
    output_root: Path,
    config: LongitudinalConfig,
    registration_run_id: str | None = None,
    localisation_run_ids: list[str] | None = None,
    segmentation_run_ids: list[str] | None = None,
    classification_run_ids: list[str] | None = None,
    upstream_quality_statuses: dict[str, str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run deterministic longitudinal analysis for one previous/current mask pair."""
    _validate_no_direct_identifier(case_id, research_subject_id)
    if side not in {"left", "right"}:
        raise LongitudinalAnalysisError("Side must be left or right.")
    if not _temporal_order_valid(previous_timepoint, current_timepoint):
        raise LongitudinalAnalysisError("Previous timepoint must sort before current timepoint.")
    previous_mask = validate_binary_mask(np.load(previous_mask_path))
    current_mask = validate_binary_mask(np.load(current_mask_path))
    geometry_compatible = previous_mask.shape == current_mask.shape and tuple(
        previous_spacing_mm
    ) == tuple(current_spacing_mm)
    if not geometry_compatible and registration_run_id is None:
        raise LongitudinalAnalysisError("Geometry mismatch requires explicit registration_run_id.")
    analysis_id = _analysis_id(case_id, side, config.policy_version)
    output_dir = _safe_output_dir(output_root, analysis_id)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Longitudinal analysis already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pair = PairManifest(
        analysis_id=analysis_id,
        case_id=case_id,
        research_subject_id=research_subject_id,
        side=side,  # type: ignore[arg-type]
        previous_timepoint=previous_timepoint,
        current_timepoint=current_timepoint,
        previous_mask=str(previous_mask_path),
        current_mask=str(current_mask_path),
        previous_spacing_mm=previous_spacing_mm,
        current_spacing_mm=current_spacing_mm,
        registration_run_id=registration_run_id,
        localisation_run_ids=localisation_run_ids or [],
        segmentation_run_ids=segmentation_run_ids or [],
        classification_run_ids=classification_run_ids or [],
        source_checksums={
            "previous_mask": sha256_file(previous_mask_path),
            "current_mask": sha256_file(current_mask_path),
        },
        upstream_quality_statuses=upstream_quality_statuses
        or {"registration": "PASS", "segmentation": "PASS"},
        generated_at=stable_timestamp(),
    )
    previous_measurements = measure_components(
        previous_mask,
        previous_spacing_mm,
        timepoint="previous",
        minimum_component_voxels=config.minimum_component_voxels,
    )
    current_measurements = measure_components(
        current_mask,
        current_spacing_mm,
        timepoint="current",
        minimum_component_voxels=config.minimum_component_voxels,
    )
    forced = forced_indeterminate_reasons(pair, config, geometry_compatible)
    matches = (
        match_lesions(
            previous_mask, current_mask, previous_measurements, current_measurements, config
        )
        if geometry_compatible
        else []
    )
    changes = calculate_changes(
        matches,
        previous_measurements,
        current_measurements,
        config,
        force_indeterminate_reasons=forced,
    )
    review_arrays = write_review_arrays(output_dir, previous_mask, current_mask)
    status, findings = evaluate_quality(
        config=config,
        pair=pair,
        previous_measurements=previous_measurements,
        current_measurements=current_measurements,
        changes=changes,
        output_paths=[output_dir / path for path in review_arrays.values()],
        geometry_compatible=geometry_compatible,
        match_ambiguous=any(match.ambiguous for match in matches),
    )
    payload: dict[str, Any] = {
        "analysis_id": analysis_id,
        "policy_version": config.policy_version,
        "status": status,
        "pair_manifest": pair.model_dump(mode="json"),
        "lesion_measurements": {
            "previous": [item.model_dump(mode="json") for item in previous_measurements],
            "current": [item.model_dump(mode="json") for item in current_measurements],
        },
        "lesion_matches": [item.model_dump(mode="json") for item in matches],
        "longitudinal_changes": [item.model_dump(mode="json") for item in changes],
        "quality_findings": [item.model_dump(mode="json") for item in findings],
        "summary": {
            "analysis_id": analysis_id,
            "status": status,
            "engineering_labels": sorted({change.label for change in changes}),
            "change_count": len(changes),
            "matched_count": sum(1 for match in matches if match.status == "matched"),
            "indeterminate_count": sum(1 for change in changes if change.label == "indeterminate"),
            "review_arrays": review_arrays,
        },
        "paths": {
            "pair_manifest": "pair_manifest.json",
            "lesion_measurements": "lesion_measurements.json",
            "lesion_matches": "lesion_matches.json",
            "longitudinal_changes": "longitudinal_changes.json",
            "quality_findings": "quality_findings.json",
            "longitudinal_summary": "longitudinal_summary.json",
            "longitudinal_report": "longitudinal_report.md",
            **review_arrays,
        },
        "recommended_next_action": _recommended_next_action(status),
    }
    checksums = write_longitudinal_evidence(output_dir, payload)
    for key, relative_path in review_arrays.items():
        checksums[key] = sha256_file(output_dir / relative_path)
    payload["checksums"] = checksums
    write_manifest(output_dir, payload)
    return payload


def load_upstream_statuses(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LongitudinalAnalysisError("Upstream status JSON must be an object.")
    return {str(key): str(value) for key, value in payload.items()}


def _analysis_id(case_id: str, side: str, policy_version: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "-" for character in case_id
    )
    return f"longitudinal-{safe}-{side}-{policy_version}"


def _safe_output_dir(root: Path, child: str) -> Path:
    resolved_root = root.resolve()
    child_path = (resolved_root / child).resolve()
    if os.path.commonpath([str(resolved_root), str(child_path)]) != str(resolved_root):
        raise LongitudinalAnalysisError("Longitudinal output path escapes configured root.")
    return child_path


def _validate_no_direct_identifier(case_id: str, research_subject_id: str) -> None:
    forbidden = ("patient", "name", "dob", "nhs", "mrn")
    combined = f"{case_id} {research_subject_id}".lower()
    if any(token in combined for token in forbidden):
        raise LongitudinalAnalysisError("Pair metadata appears to contain direct identifiers.")


def _temporal_order_valid(previous_timepoint: str, current_timepoint: str) -> bool:
    ordered = {
        "baseline": 0,
        "prior": 0,
        "previous": 0,
        "followup": 1,
        "current": 1,
    }
    previous = ordered.get(previous_timepoint.lower())
    current = ordered.get(current_timepoint.lower())
    if previous is not None and current is not None:
        return previous < current
    return previous_timepoint < current_timepoint


def _recommended_next_action(status: str) -> str:
    if status == "PASS":
        return "Proceed only to research engineering review and Milestone 11 planning."
    if status == "PASS_WITH_WARNINGS":
        return "Review longitudinal warnings before downstream research use."
    return "Do not use longitudinal labels downstream without remediation."
