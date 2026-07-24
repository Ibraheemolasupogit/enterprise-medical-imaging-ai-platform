"""Quality gates for longitudinal synthetic lesion analysis."""

from __future__ import annotations

from pathlib import Path

from medical_imaging_platform.longitudinal.models import (
    LesionMeasurement,
    LongitudinalChange,
    LongitudinalConfig,
    LongitudinalFinding,
    LongitudinalStatus,
    PairManifest,
)


def evaluate_quality(
    *,
    config: LongitudinalConfig,
    pair: PairManifest,
    previous_measurements: list[LesionMeasurement],
    current_measurements: list[LesionMeasurement],
    changes: list[LongitudinalChange],
    output_paths: list[Path],
    side_consistent: bool = True,
    temporal_valid: bool = True,
    geometry_compatible: bool = True,
    match_ambiguous: bool = False,
) -> tuple[LongitudinalStatus, list[LongitudinalFinding]]:
    """Evaluate deterministic longitudinal quality gates."""
    measurements_finite = _finite_measurements(previous_measurements + current_measurements)
    findings = [
        _finding("LNG-QC-PAIR-001", "INFO", "Pair metadata is present.", "PASS"),
        _finding(
            "LNG-QC-TIME-001",
            "INFO" if temporal_valid else "CRITICAL",
            "Temporal ordering is previous before current."
            if temporal_valid
            else "Temporal ordering is invalid.",
            "PASS" if temporal_valid else "FAIL",
        ),
        _finding(
            "LNG-QC-SIDE-001",
            "INFO" if side_consistent else "CRITICAL",
            "Anatomical side is consistent."
            if side_consistent
            else "Anatomical side mismatch detected.",
            "PASS" if side_consistent else "FAIL",
        ),
        _finding(
            "LNG-QC-GEO-001",
            "INFO" if geometry_compatible else "ERROR",
            "Geometry is compatible or registered-space mapping is explicit."
            if geometry_compatible
            else "Geometry is incompatible without an explicit registered-space mapping.",
            "PASS" if geometry_compatible else "FAIL",
        ),
        _upstream("LNG-QC-REG-001", "registration", pair, config.require_registration_pass),
        _upstream("LNG-QC-SEG-001", "segmentation", pair, config.require_segmentation_pass),
        _finding(
            "LNG-QC-MATCH-001",
            "WARNING" if match_ambiguous else "INFO",
            "Ambiguous match detected." if match_ambiguous else "Lesion matching completed.",
            "FAIL" if match_ambiguous else "PASS",
        ),
        _finding(
            "LNG-QC-MEAS-001",
            "INFO" if measurements_finite else "ERROR",
            "Measurements are finite.",
            "PASS" if measurements_finite else "FAIL",
        ),
        _finding(
            "LNG-QC-CHANGE-001",
            "INFO" if changes else "ERROR",
            "Longitudinal change records were generated.",
            "PASS" if changes else "FAIL",
        ),
        _finding(
            "LNG-QC-LABEL-001",
            "INFO" if all(change.label for change in changes) else "ERROR",
            "Engineering labels are present.",
            "PASS" if all(change.label for change in changes) else "FAIL",
        ),
        _finding(
            "LNG-QC-PROV-001",
            "INFO" if pair.source_checksums and pair.upstream_quality_statuses else "ERROR",
            "Provenance and upstream quality statuses are recorded.",
            "PASS" if pair.source_checksums and pair.upstream_quality_statuses else "FAIL",
        ),
        _finding(
            "LNG-QC-CHK-001",
            "INFO" if all(path.exists() for path in output_paths) else "ERROR",
            "Output evidence files are present.",
            "PASS" if all(path.exists() for path in output_paths) else "FAIL",
        ),
    ]
    if any(finding.severity == "CRITICAL" and finding.status == "FAIL" for finding in findings):
        return "REJECTED", findings
    if any(finding.severity == "ERROR" and finding.status == "FAIL" for finding in findings):
        return "FAIL", findings
    if any(finding.status == "FAIL" for finding in findings):
        return "PASS_WITH_WARNINGS", findings
    return "PASS", findings


def forced_indeterminate_reasons(
    pair: PairManifest, config: LongitudinalConfig, geometry_compatible: bool
) -> list[str]:
    """Return upstream reasons that force indeterminate longitudinal labels."""
    reasons: list[str] = []
    if config.require_registration_pass and pair.upstream_quality_statuses.get(
        "registration"
    ) not in {"PASS", "PASS_WITH_WARNINGS"}:
        reasons.append("registration quality did not pass")
    if config.require_segmentation_pass and pair.upstream_quality_statuses.get(
        "segmentation"
    ) not in {"PASS", "PASS_WITH_WARNINGS"}:
        reasons.append("segmentation quality did not pass")
    if (
        config.classification_abstention_forces_indeterminate
        and pair.upstream_quality_statuses.get("classification_abstention") == "ABSTAINED"
    ):
        reasons.append("classification abstention propagated")
    if not geometry_compatible:
        reasons.append("geometry compatibility failed")
    return reasons


def _upstream(rule_id: str, key: str, pair: PairManifest, required: bool) -> LongitudinalFinding:
    status = pair.upstream_quality_statuses.get(key, "MISSING")
    ok = (not required and status == "MISSING") or status in {"PASS", "PASS_WITH_WARNINGS"}
    return _finding(
        rule_id,
        "INFO" if ok else "ERROR",
        f"{key} upstream status is {status}.",
        "PASS" if ok else "FAIL",
        observed=status,
        expected="PASS or PASS_WITH_WARNINGS" if required else "optional",
    )


def _finite_measurements(measurements: list[LesionMeasurement]) -> bool:
    for measurement in measurements:
        values = [
            measurement.physical_volume_mm3,
            measurement.physical_volume_ml,
            measurement.maximum_3d_diameter_mm,
            measurement.axial_maximum_diameter_mm,
        ]
        if any(value != value or value in {float("inf"), float("-inf")} for value in values):
            return False
    return True


def _finding(
    rule_id: str,
    severity: str,
    message: str,
    status: str,
    observed: object | None = None,
    expected: object | None = None,
) -> LongitudinalFinding:
    return LongitudinalFinding(
        rule_id=rule_id,
        severity=severity,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        message=message,
        observed_value=observed,
        expected_value=expected,
        remediation=(
            "Review pair metadata, upstream quality, masks, spacing, matching, and thresholds."
        ),
    )
