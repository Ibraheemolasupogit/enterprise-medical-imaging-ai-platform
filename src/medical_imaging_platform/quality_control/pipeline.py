"""DICOM quality-control orchestration."""

from __future__ import annotations

from pathlib import Path

from medical_imaging_platform.ingestion.discovery import discover_dicom_series
from medical_imaging_platform.ingestion.loader import load_dicom
from medical_imaging_platform.ingestion.metadata import extract_metadata
from medical_imaging_platform.ingestion.models import DicomFileMetadata
from medical_imaging_platform.ingestion.ordering import order_slices
from medical_imaging_platform.quality_control.metadata_checks import run_metadata_checks
from medical_imaging_platform.quality_control.models import (
    QualityControlConfig,
    QualityFinding,
    QualityReport,
)
from medical_imaging_platform.quality_control.pixel_checks import run_pixel_checks
from medical_imaging_platform.quality_control.report import write_quality_reports
from medical_imaging_platform.quality_control.rules import rule_info
from medical_imaging_platform.quality_control.scoring import score_findings
from medical_imaging_platform.quality_control.slice_checks import run_slice_checks


def run_quality_control(
    input_dir: Path,
    *,
    output_dir: Path | None,
    qc_config: QualityControlConfig,
    max_files: int,
    max_file_size_bytes: int,
    full_pixel_validation: bool = False,
    overwrite: bool = False,
) -> list[QualityReport]:
    """Run quality control for every discovered series."""
    discovery = discover_dicom_series(
        input_dir,
        max_files=max_files,
        max_file_size_bytes=max_file_size_bytes,
    )
    reports: list[QualityReport] = []
    skipped_findings = [
        _finding(
            "DICOM-QC-FILE-001",
            "FAIL",
            f"Skipped file: {skipped.reason}",
            [skipped.path],
            skipped.reason,
            "readable DICOM file within configured limits",
            "Review skipped file or discovery limits.",
        )
        for skipped in discovery.skipped_files
    ]
    for index, series in enumerate(discovery.series):
        metadata: list[DicomFileMetadata] = []
        file_findings: list[QualityFinding] = list(skipped_findings)
        for file_path in series.files:
            try:
                dataset = load_dicom(
                    Path(file_path),
                    header_only=not full_pixel_validation,
                    max_file_size_bytes=max_file_size_bytes,
                )
                metadata.append(extract_metadata(dataset, Path(file_path)))
            except Exception as exc:
                file_findings.append(
                    _finding(
                        "DICOM-QC-FILE-001",
                        "FAIL",
                        "DICOM file could not be loaded.",
                        [file_path],
                        str(exc),
                        "readable DICOM file",
                        "Review source file.",
                    )
                )
        report = build_quality_report(
            metadata,
            file_findings=file_findings,
            qc_config=qc_config,
            max_file_size_bytes=max_file_size_bytes,
            full_pixel_validation=full_pixel_validation,
            report_index=index,
        )
        reports.append(report)
        if output_dir is not None:
            series_output_dir = (
                output_dir if len(discovery.series) == 1 else output_dir / f"series-{index + 1:03d}"
            )
            write_quality_reports(report, series_output_dir, overwrite=overwrite)
    if not discovery.series and output_dir is not None:
        report = build_quality_report(
            [],
            file_findings=skipped_findings,
            qc_config=qc_config,
            max_file_size_bytes=max_file_size_bytes,
            full_pixel_validation=full_pixel_validation,
            report_index=0,
        )
        reports.append(report)
        write_quality_reports(report, output_dir, overwrite=overwrite)
    return reports


def build_quality_report(
    metadata: list[DicomFileMetadata],
    *,
    file_findings: list[QualityFinding],
    qc_config: QualityControlConfig,
    max_file_size_bytes: int,
    full_pixel_validation: bool,
    report_index: int = 0,
) -> QualityReport:
    """Build one series-level quality report."""
    ordered = order_slices(metadata)
    metadata_findings, metadata_metrics = (
        run_metadata_checks(metadata, qc_config) if metadata else ([], {})
    )
    slice_findings, slice_metrics = run_slice_checks(ordered, qc_config) if metadata else ([], {})
    pixel_findings: list[QualityFinding] = []
    pixel_metrics: dict[str, object] = {"pixel_validation_mode": "header_only"}
    if full_pixel_validation and metadata:
        pixel_findings, pixel_metrics = run_pixel_checks(
            metadata,
            qc_config,
            max_file_size_bytes=max_file_size_bytes,
        )
        pixel_metrics["pixel_validation_mode"] = "full"
    findings = file_findings + metadata_findings + slice_findings + pixel_findings
    score, status, deductions = score_findings(findings, qc_config)
    first = metadata[0] if metadata else None
    return QualityReport(
        quality_report_id=f"dicom-qc-report-{report_index + 1:04d}",
        study_instance_uid=first.study_instance_uid if first else None,
        series_instance_uid=first.series_instance_uid if first else None,
        status=status,
        quality_score=score,
        evaluated_file_count=len(metadata),
        expected_slice_count=_expected_slice_count(slice_metrics),
        observed_slice_count=len(metadata),
        ordering_strategy=ordered.strategy,
        findings=sorted(findings, key=lambda item: (item.severity, item.rule_id, item.message)),
        metrics={
            "metadata": metadata_metrics,
            "slice_completeness": slice_metrics,
            "pixel_integrity": pixel_metrics,
        },
        generated_at="2026-01-01T00:00:00+00:00",
        policy_version=qc_config.policy_version,
        score_deductions=deductions,
    )


def _expected_slice_count(slice_metrics: dict[str, object]) -> int | None:
    observed = slice_metrics.get("observed_slice_count")
    irregular = slice_metrics.get("irregular_spacing_count")
    if isinstance(observed, int) and irregular == 0:
        return observed
    return None


def _finding(
    rule_id: str,
    status: str,
    message: str,
    affected_files: list[str],
    observed_value: object,
    expected_value: object,
    remediation: str,
) -> QualityFinding:
    category, severity, _ = rule_info(rule_id)
    return QualityFinding(
        rule_id=rule_id,
        category=category,
        severity=severity,
        status=status,  # type: ignore[arg-type]
        message=message,
        affected_files=affected_files,
        observed_value=observed_value,
        expected_value=expected_value,
        remediation=remediation,
    )
