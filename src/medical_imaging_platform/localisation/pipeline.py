"""Atlas-style localisation pipeline."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from medical_imaging_platform.localisation.atlas import centre_mm, predict_centre_voxel
from medical_imaging_platform.localisation.export import (
    build_output_paths,
    write_localisation_outputs,
)
from medical_imaging_platform.localisation.metrics import evaluate_side_metrics
from medical_imaging_platform.localisation.models import (
    LocalisationConfig,
    LocalisationFinding,
    LocalisationMetrics,
    LocalisationResult,
    Side,
    SideLocalisation,
    SourceVolumeMetadata,
)
from medical_imaging_platform.localisation.quality import evaluate_quality
from medical_imaging_platform.localisation.roi import (
    bounding_box_mm,
    extract_roi,
    overlay_mid_slice,
)
from medical_imaging_platform.preprocessing.export import validate_preprocessed_volume
from medical_imaging_platform.registration.export import validate_registration_output


def localise_adrenal_regions(
    input_dir: Path,
    *,
    output_root: Path,
    config: LocalisationConfig,
    mode: str | None = None,
    left_mask_path: Path | None = None,
    right_mask_path: Path | None = None,
    overwrite: bool | None = None,
) -> LocalisationResult:
    """Run deterministic atlas localisation on one explicit input volume."""
    selected_mode = mode or config.default_mode
    if selected_mode != "atlas":
        raise ValueError(f"Unsupported localisation mode: {selected_mode}")
    volume, source, findings = load_source_volume(input_dir)
    left_mask = _load_mask(left_mask_path, volume.shape)
    right_mask = _load_mask(right_mask_path, volume.shape)
    spacing = source.spacing_mm_zyx
    left_result, left_roi, left_overlay = _localise_side("left", volume, source, config)
    right_result, right_roi, right_overlay = _localise_side("right", volume, source, config)
    metrics: dict[Side, LocalisationMetrics] = {
        "left": evaluate_side_metrics(
            "left",
            left_result.predicted_centre_voxel,
            left_result.bounding_box_voxel,
            spacing,
            left_mask=left_mask,
            right_mask=right_mask,
        ),
        "right": evaluate_side_metrics(
            "right",
            right_result.predicted_centre_voxel,
            right_result.bounding_box_voxel,
            spacing,
            left_mask=left_mask,
            right_mask=right_mask,
        ),
    }
    overall_status, findings = evaluate_quality(
        left_result,
        right_result,
        metrics,
        spacing_mm_zyx=spacing,
        config=config,
        input_findings=findings,
    )
    if overall_status == "REJECTED":
        left_result = left_result.model_copy(update={"status": overall_status})
        right_result = right_result.model_copy(update={"status": overall_status})
    run_id = _run_id(source.source_run_id, selected_mode, config.policy_version)
    result = LocalisationResult(
        localisation_run_id=run_id,
        source=source,
        localisation_mode="atlas",
        configuration_version=config.policy_version,
        left=left_result,
        right=right_result,
        ground_truth_available=left_mask is not None and right_mask is not None,
        metrics=metrics,
        quality_findings=findings,
        overall_status=overall_status,
        warnings=sorted({finding.message for finding in findings if finding.severity == "WARNING"}),
        output_paths=build_output_paths(output_root, run_id),
        checksums={},
        generated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        limitations=[
            "Atlas/geometry baseline only.",
            "Synthetic engineering labels are not clinical adrenal annotations.",
            "No learned model, organ segmentation, lesion detection, or classification exists.",
        ],
        recommended_next_action=_recommended_next_action(overall_status),
    )
    return write_localisation_outputs(
        left_roi,
        right_roi,
        left_overlay,
        right_overlay,
        result,
        overwrite=config.overwrite if overwrite is None else overwrite,
    )


def load_source_volume(
    input_dir: Path,
) -> tuple[np.ndarray, SourceVolumeMetadata, list[LocalisationFinding]]:
    """Load either a preprocessing output or registration output for localisation."""
    findings: list[LocalisationFinding] = []
    if (input_dir / "registration_metadata.json").exists():
        reg = validate_registration_output(input_dir)
        if reg.status not in {"PASS", "PASS_WITH_WARNINGS"}:
            findings.append(
                _finding("LOC-QC-INP-001", "CRITICAL", "Registration status is not acceptable.")
            )
        volume = np.load(reg.output_paths.registered_moving_volume)
        source = SourceVolumeMetadata(
            source_type="registration",
            source_dir=str(input_dir),
            source_run_id=reg.registration_run_id,
            source_status=reg.status,
            upstream_override_used=reg.fixed.preprocessing_override_used
            or reg.moving.preprocessing_override_used,
            volume_shape=reg.fixed.volume_shape,
            spacing_mm_zyx=reg.fixed.spacing_mm_zyx,
            axis_order=reg.fixed.axis_order,
        )
    else:
        prep = validate_preprocessed_volume(input_dir)
        volume = np.load(prep.output_paths.volume)
        if prep.source_quality_status == "REJECTED":
            findings.append(
                _finding("LOC-QC-INP-001", "CRITICAL", "Source quality status is REJECTED.")
            )
        source = SourceVolumeMetadata(
            source_type="preprocessing",
            source_dir=str(input_dir),
            source_run_id=prep.run_id,
            source_status=prep.source_quality_status,
            upstream_override_used=prep.override_used,
            volume_shape=prep.volume_shape,
            spacing_mm_zyx=prep.spacing_mm,
            axis_order=prep.axis_order,
        )
    findings.extend(_input_findings(volume, source))
    return volume, source, findings


def _localise_side(
    side: Side,
    volume: np.ndarray,
    source: SourceVolumeMetadata,
    config: LocalisationConfig,
) -> tuple[SideLocalisation, np.ndarray, np.ndarray]:
    centre = predict_centre_voxel(
        side,
        volume_shape=source.volume_shape,
        spacing_mm_zyx=source.spacing_mm_zyx,
        config=config,
    )
    roi, extraction = extract_roi(
        volume, centre=centre, spacing_mm_zyx=source.spacing_mm_zyx, config=config
    )
    box = extraction.crop_bounds_zyx
    overlay = overlay_mid_slice(volume, box, centre)
    confidence = max(0.0, 1.0 - extraction.padding_fraction)
    warnings = ["ROI required boundary padding."] if extraction.padding_fraction > 0 else []
    return (
        SideLocalisation(
            side=side,
            predicted_centre_voxel=centre,
            predicted_centre_mm=centre_mm(centre, source.spacing_mm_zyx),
            bounding_box_voxel=box,
            bounding_box_mm=bounding_box_mm(box, source.spacing_mm_zyx),
            roi_shape=extraction.roi_shape,
            confidence=confidence,
            status="LOCALISED_WITH_WARNINGS" if warnings else "LOCALISED",
            warnings=warnings,
            roi_extraction=extraction,
        ),
        roi,
        overlay,
    )


def _input_findings(volume: np.ndarray, source: SourceVolumeMetadata) -> list[LocalisationFinding]:
    findings: list[LocalisationFinding] = []
    if volume.ndim != 3:
        findings.append(_finding("LOC-QC-INP-001", "CRITICAL", "Input volume is not 3D."))
    if not np.all(np.isfinite(volume)):
        findings.append(
            _finding("LOC-QC-INP-001", "CRITICAL", "Input volume contains non-finite values.")
        )
    if source.axis_order != "z,y,x":
        findings.append(_finding("LOC-QC-GEO-001", "CRITICAL", "Input axis order is not z,y,x."))
    if any(item <= 0 or not np.isfinite(item) for item in source.spacing_mm_zyx):
        findings.append(_finding("LOC-QC-GEO-001", "CRITICAL", "Input spacing is invalid."))
    if source.upstream_override_used:
        findings.append(_finding("LOC-QC-OVR-001", "WARNING", "Upstream override was propagated."))
    return findings


def _load_mask(path: Path | None, shape: tuple[int, ...]) -> np.ndarray | None:
    if path is None:
        return None
    mask = np.asarray(np.load(path), dtype=bool)
    if tuple(mask.shape) != tuple(shape):
        raise ValueError("Ground-truth mask shape does not match source volume.")
    return mask


def _finding(rule_id: str, severity: str, message: str) -> LocalisationFinding:
    return LocalisationFinding(
        rule_id=rule_id,
        severity=severity,  # type: ignore[arg-type]
        status="FAIL",
        message=message,
        remediation="Review localisation input and upstream provenance.",
    )


def _run_id(source_run_id: str, mode: str, policy_version: str) -> str:
    digest = hashlib.sha256(
        "|".join([source_run_id, mode, policy_version]).encode("utf-8")
    ).hexdigest()[:16]
    return f"localisation-{digest}"


def _recommended_next_action(status: str) -> str:
    if status == "LOCALISED":
        return "Proceed only to research engineering review."
    if status == "LOCALISED_WITH_WARNINGS":
        return "Review warnings before any downstream research use."
    return "Do not use this localisation output downstream without remediation."
