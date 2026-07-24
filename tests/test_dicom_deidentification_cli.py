import json
from pathlib import Path

import pydicom
import pytest

from medical_imaging_platform.cli import main
from medical_imaging_platform.deidentification.deidentifier import (
    DeidentificationError,
    deidentify_series,
)
from medical_imaging_platform.deidentification.policy import default_policy
from medical_imaging_platform.ingestion.fixtures import generate_dicom_fixture_series


def policy():
    return default_policy(
        policy_version="test-policy",
        uid_root="1.2.826.0.1.3680043.10.54321.99",
        patient_id_prefix="RSP",
    )


def test_deidentification_removes_identifiers_private_tags_and_remaps_uids(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    audit_path = tmp_path / "audit" / "audit.json"
    paths = generate_dicom_fixture_series(source, slice_count=2)
    original = pydicom.dcmread(paths[0], stop_before_pixels=True)

    audit = deidentify_series(
        paths,
        output_dir=output,
        audit_path=audit_path,
        policy=policy(),
        overwrite=True,
        max_file_size_bytes=1_000_000,
    )
    deidentified = pydicom.dcmread(output / "deidentified-001.dcm", stop_before_pixels=True)

    assert audit.file_count == 2
    assert deidentified.PatientName != "Synthetic^Fixture"
    assert deidentified.PatientID.startswith("RSP-")
    assert "AccessionNumber" not in deidentified
    assert deidentified.StudyInstanceUID != original.StudyInstanceUID
    assert sum(1 for element in deidentified.iterall() if element.tag.is_private) == 0
    assert "Synthetic^Fixture" not in audit_path.read_text(encoding="utf-8")


def test_burned_in_annotation_handling(tmp_path: Path) -> None:
    source = tmp_path / "source"
    paths = generate_dicom_fixture_series(source, slice_count=1, malformed="burned_in")

    audit = deidentify_series(
        paths,
        output_dir=tmp_path / "output",
        audit_path=tmp_path / "audit.json",
        policy=policy(),
        overwrite=True,
        max_file_size_bytes=1_000_000,
    )

    assert audit.files[0].burned_in_annotation_status == "YES"
    assert audit.files[0].warnings


def test_no_source_overwrite_and_explicit_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    paths = generate_dicom_fixture_series(source, slice_count=1)
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("exists", encoding="utf-8")

    with pytest.raises(DeidentificationError, match="not empty"):
        deidentify_series(
            paths,
            output_dir=output,
            audit_path=tmp_path / "audit.json",
            policy=policy(),
            overwrite=False,
            max_file_size_bytes=1_000_000,
        )

    audit = deidentify_series(
        paths,
        output_dir=output,
        audit_path=tmp_path / "audit.json",
        policy=policy(),
        overwrite=True,
        max_file_size_bytes=1_000_000,
    )
    assert audit.file_count == 1


def test_cli_dicom_fixture_discovery_inspection_validation_and_deidentification(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    fixture_dir = tmp_path / "fixtures"
    output_dir = tmp_path / "deid"
    audit_path = tmp_path / "audit.json"

    assert (
        main(["generate-dicom-fixtures", "--output-dir", str(fixture_dir), "--slice-count", "2"])
        == 0
    )
    assert "Generated 2 DICOM fixture files" in capsys.readouterr().out

    assert main(["discover-dicom", str(fixture_dir), "--json"]) == 0
    discovery = json.loads(capsys.readouterr().out)
    assert len(discovery["series"]) == 1

    first_file = fixture_dir / "slice-001.dcm"
    assert main(["inspect-dicom", str(first_file), "--json"]) == 0
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["modality"] == "CT"
    assert "Synthetic^Fixture" not in json.dumps(metadata)

    assert main(["validate-dicom", str(fixture_dir), "--require-pixel-data"]) == 0
    assert "0 findings" in capsys.readouterr().out

    assert (
        main(
            [
                "deidentify-dicom",
                str(fixture_dir),
                "--output-dir",
                str(output_dir),
                "--audit-path",
                str(audit_path),
            ]
        )
        == 0
    )
    assert "De-identified 2 files" in capsys.readouterr().out
    assert audit_path.exists()


def test_cli_error_exit_codes_and_overwrite_protection(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["discover-dicom", str(tmp_path / "missing")]) == 1
    assert "DICOM discovery failed" in capsys.readouterr().out

    fixture_dir = tmp_path / "fixtures"
    assert main(["generate-dicom-fixtures", "--output-dir", str(fixture_dir)]) == 0
    assert main(["generate-dicom-fixtures", "--output-dir", str(fixture_dir)]) == 1
    assert "not empty" in capsys.readouterr().out
