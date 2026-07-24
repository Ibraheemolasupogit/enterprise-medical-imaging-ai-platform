import json
from pathlib import Path

import pydicom
import pytest
from pydantic import ValidationError

from medical_imaging_platform.cli import main
from medical_imaging_platform.deidentification.deidentifier import deidentify_series
from medical_imaging_platform.deidentification.policy import default_policy
from medical_imaging_platform.ingestion.fixtures import generate_dicom_fixture_series
from medical_imaging_platform.ingestion.loader import load_dicom
from medical_imaging_platform.ingestion.metadata import extract_metadata
from medical_imaging_platform.quality_control.models import QualityControlConfig, QualityFinding
from medical_imaging_platform.quality_control.pipeline import (
    build_quality_report,
    run_quality_control,
)
from medical_imaging_platform.quality_control.report import write_quality_reports
from medical_imaging_platform.quality_control.rules import RULE_CATALOGUE
from medical_imaging_platform.quality_control.scoring import score_findings


def qc_config(tmp_path: Path) -> QualityControlConfig:
    return QualityControlConfig(
        policy_version="test-qc",
        output_dir=tmp_path / "quality",
        accepted_modalities=["CT"],
        body_region_allowlist=["ABDOMEN"],
        require_body_part_examined=True,
        pixel_spacing_tolerance=0.001,
        slice_thickness_tolerance=0.001,
        orientation_tolerance=0.001,
        slice_gap_tolerance=0.01,
        minimum_slice_count=2,
        maximum_slice_count=100,
        maximum_corrupt_file_ratio=0.0,
        burned_in_annotation_policy="reject_yes_warn_missing",
        private_tag_policy="warn_before_deidentification_error_after",
        supported_transfer_syntaxes=["1.2.840.10008.1.2.1"],
        pixel_value_bounds=(-4096, 4096),
        scoring_weights={"INFO": 0, "WARNING": 5, "ERROR": 25, "CRITICAL": 100},
        status_thresholds={"pass_minimum": 95, "fail_below": 70},
        critical_rule_ids=["DICOM-QC-PHI-001"],
    )


def metadata_for(paths: list[Path]):
    return [
        extract_metadata(load_dicom(path, header_only=True, max_file_size_bytes=1_000_000), path)
        for path in paths
    ]


def deidentified_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    output = tmp_path / "deid"
    paths = generate_dicom_fixture_series(source, slice_count=3)
    deidentify_series(
        paths,
        output_dir=output,
        audit_path=tmp_path / "audit.json",
        policy=default_policy(
            policy_version="test",
            uid_root="1.2.826.0.1.3680043.10.54321.88",
            patient_id_prefix="RSP",
        ),
        overwrite=True,
        max_file_size_bytes=1_000_000,
    )
    return output


def test_rule_model_values_and_catalogue_are_stable(tmp_path: Path) -> None:
    finding = QualityFinding(
        rule_id="DICOM-QC-MOD-001",
        category="METADATA",
        severity="ERROR",
        status="FAIL",
        message="bad modality",
        remediation="Use CT",
    )

    assert finding.model_dump()["rule_id"] == "DICOM-QC-MOD-001"
    assert "DICOM-QC-PIX-004" in RULE_CATALOGUE
    with pytest.raises(ValidationError):
        QualityFinding(
            rule_id="bad",
            category="METADATA",
            severity="BAD",  # type: ignore[arg-type]
            status="FAIL",
            message="bad",
            remediation="fix",
        )


def test_valid_header_only_deidentified_series_passes(tmp_path: Path) -> None:
    output = deidentified_fixture(tmp_path)

    reports = run_quality_control(
        output,
        output_dir=None,
        qc_config=qc_config(tmp_path),
        max_files=20,
        max_file_size_bytes=1_000_000,
    )

    assert reports[0].status == "PASS"
    assert reports[0].quality_score == 100
    assert "Synthetic^Fixture" not in reports[0].model_dump_json()


def test_full_pixel_validation_reports_constant_slice_warning(tmp_path: Path) -> None:
    output = deidentified_fixture(tmp_path)

    reports = run_quality_control(
        output,
        output_dir=None,
        qc_config=qc_config(tmp_path),
        max_files=20,
        max_file_size_bytes=1_000_000,
        full_pixel_validation=True,
    )

    assert reports[0].status == "PASS_WITH_WARNINGS"
    assert "DICOM-QC-PIX-004" in {finding.rule_id for finding in reports[0].findings}


def test_slice_completeness_duplicate_and_irregular_spacing(tmp_path: Path) -> None:
    paths = generate_dicom_fixture_series(tmp_path, slice_count=3)
    duplicate = pydicom.dcmread(paths[1])
    duplicate.ImagePositionPatient = [0.0, 0.0, 0.0]
    duplicate.InstanceNumber = 1
    duplicate.save_as(paths[1])
    irregular = pydicom.dcmread(paths[2])
    irregular.ImagePositionPatient = [0.0, 0.0, 10.0]
    irregular.save_as(paths[2])

    report = build_quality_report(
        metadata_for(paths),
        file_findings=[],
        qc_config=qc_config(tmp_path),
        max_file_size_bytes=1_000_000,
        full_pixel_validation=False,
    )
    rule_ids = {finding.rule_id for finding in report.findings}

    assert "DICOM-QC-SLC-001" in rule_ids
    assert "DICOM-QC-SLC-003" in rule_ids
    assert "DICOM-QC-SLC-002" in rule_ids


def test_metadata_rules_wrong_modality_body_region_spacing_and_orientation(tmp_path: Path) -> None:
    paths = generate_dicom_fixture_series(tmp_path, slice_count=2)
    dataset = pydicom.dcmread(paths[1])
    dataset.Modality = "MR"
    dataset.BodyPartExamined = "HEAD"
    dataset.PixelSpacing = [3.0, 3.0]
    dataset.ImageOrientationPatient = [0, 1, 0, 1, 0, 0]
    dataset.save_as(paths[1])

    report = build_quality_report(
        metadata_for(paths),
        file_findings=[],
        qc_config=qc_config(tmp_path),
        max_file_size_bytes=1_000_000,
        full_pixel_validation=False,
    )
    rule_ids = {finding.rule_id for finding in report.findings}

    assert "DICOM-QC-MOD-001" in rule_ids
    assert "DICOM-QC-BODY-001" in rule_ids
    assert "DICOM-QC-SPC-002" in rule_ids
    assert "DICOM-QC-ORI-002" in rule_ids


def test_burned_in_annotation_critical_rejection(tmp_path: Path) -> None:
    paths = generate_dicom_fixture_series(tmp_path, slice_count=2, malformed="burned_in")

    report = build_quality_report(
        metadata_for(paths),
        file_findings=[],
        qc_config=qc_config(tmp_path),
        max_file_size_bytes=1_000_000,
        full_pixel_validation=False,
    )

    assert report.status == "REJECTED"
    assert report.quality_score == 0


def test_private_tags_warn_before_deidentification(tmp_path: Path) -> None:
    paths = generate_dicom_fixture_series(tmp_path, slice_count=2)

    report = build_quality_report(
        metadata_for(paths),
        file_findings=[],
        qc_config=qc_config(tmp_path),
        max_file_size_bytes=1_000_000,
        full_pixel_validation=False,
    )

    assert report.status == "PASS_WITH_WARNINGS"
    assert "DICOM-QC-PRV-001" in {finding.rule_id for finding in report.findings}


def test_scoring_is_bounded_traceable_and_critical_rejects(tmp_path: Path) -> None:
    config = qc_config(tmp_path)
    finding = QualityFinding(
        rule_id="DICOM-QC-PHI-001",
        category="SECURITY",
        severity="CRITICAL",
        status="FAIL",
        message="burned in",
        remediation="reject",
    )

    score, status, deductions = score_findings([finding], config)

    assert score == 0
    assert status == "REJECTED"
    assert deductions["DICOM-QC-PHI-001"] == 100


def test_reports_are_deterministic_and_exclude_identifiers(tmp_path: Path) -> None:
    output = deidentified_fixture(tmp_path)
    report = run_quality_control(
        output,
        output_dir=None,
        qc_config=qc_config(tmp_path),
        max_files=20,
        max_file_size_bytes=1_000_000,
    )[0]
    report_dir = tmp_path / "reports"

    write_quality_reports(report, report_dir)
    first_json = (report_dir / "quality_report.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_quality_reports(report, report_dir)
    write_quality_reports(report, report_dir, overwrite=True)
    second_json = (report_dir / "quality_report.json").read_text(encoding="utf-8")

    assert first_json == second_json
    assert (report_dir / "quality_report.md").exists()
    assert "Synthetic^Fixture" not in first_json


def test_cli_quality_json_fail_on_warning_and_critical_exit(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    output = deidentified_fixture(tmp_path)

    assert main(["quality-check-dicom", str(output), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["status"] == "PASS"

    assert (
        main(["quality-check-dicom", str(output), "--full-pixel-validation", "--fail-on-warning"])
        == 2
    )
    assert "PASS_WITH_WARNINGS" in capsys.readouterr().out

    burned = tmp_path / "burned"
    generate_dicom_fixture_series(burned, slice_count=2, malformed="burned_in")
    assert main(["quality-check-dicom", str(burned)]) == 3


def test_cli_quality_report_missing_input_and_overwrite_protection(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    output = deidentified_fixture(tmp_path)
    report_dir = tmp_path / "quality"

    assert main(["quality-report-dicom", str(output), "--output-dir", str(report_dir)]) == 0
    assert main(["quality-report-dicom", str(output), "--output-dir", str(report_dir)]) == 1
    assert "already exists" in capsys.readouterr().out
    assert main(["quality-check-dicom", str(tmp_path / "missing")]) == 1
