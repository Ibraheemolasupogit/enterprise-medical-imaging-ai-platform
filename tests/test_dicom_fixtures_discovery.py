from pathlib import Path

import pydicom
import pytest

from medical_imaging_platform.ingestion.discovery import discover_dicom_series
from medical_imaging_platform.ingestion.fixtures import generate_dicom_fixture_series


def test_deterministic_dicom_fixture_generation(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    study_uid = "1.2.826.0.1.3680043.10.54321.3.1.1"
    series_uid = "1.2.826.0.1.3680043.10.54321.3.2.1"

    generate_dicom_fixture_series(first, slice_count=2, study_uid=study_uid, series_uid=series_uid)
    generate_dicom_fixture_series(second, slice_count=2, study_uid=study_uid, series_uid=series_uid)

    first_ds = pydicom.dcmread(first / "slice-001.dcm", stop_before_pixels=True)
    second_ds = pydicom.dcmread(second / "slice-001.dcm", stop_before_pixels=True)

    assert first_ds.StudyInstanceUID == second_ds.StudyInstanceUID
    assert first_ds.SeriesInstanceUID == second_ds.SeriesInstanceUID
    assert first_ds.Modality == "CT"
    assert first_ds.PatientID == "SYNTHETIC-PATIENT"


def test_configurable_slice_count_and_valid_uids(tmp_path: Path) -> None:
    paths = generate_dicom_fixture_series(tmp_path, slice_count=3)

    assert len(paths) == 3
    for path in paths:
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        assert dataset.StudyInstanceUID.is_valid
        assert dataset.SeriesInstanceUID.is_valid
        assert dataset.SOPInstanceUID.is_valid


def test_malformed_fixture_for_negative_tests(tmp_path: Path) -> None:
    paths = generate_dicom_fixture_series(tmp_path, slice_count=1, malformed="wrong_modality")
    dataset = pydicom.dcmread(paths[0], stop_before_pixels=True)

    assert dataset.Modality == "MR"


def test_discovery_recursive_non_dicom_corrupt_and_stable_grouping(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    generate_dicom_fixture_series(nested, slice_count=2)
    (tmp_path / "note.txt").write_text("not dicom", encoding="utf-8")
    (tmp_path / "corrupt.dcm").write_bytes(b"not a dicom")

    result = discover_dicom_series(tmp_path, max_files=10, max_file_size_bytes=1_000_000)

    assert len(result.series) == 1
    assert len(result.series[0].files) == 2
    assert len(result.skipped_files) == 2
    assert result.series[0].files == sorted(result.series[0].files)


def test_discovery_multiple_studies_and_series(tmp_path: Path) -> None:
    generate_dicom_fixture_series(
        tmp_path / "a",
        slice_count=1,
        study_uid="1.2.826.0.1.3680043.10.54321.3.1.10",
        series_uid="1.2.826.0.1.3680043.10.54321.3.2.10",
    )
    generate_dicom_fixture_series(
        tmp_path / "b",
        slice_count=1,
        study_uid="1.2.826.0.1.3680043.10.54321.3.1.11",
        series_uid="1.2.826.0.1.3680043.10.54321.3.2.11",
    )

    result = discover_dicom_series(tmp_path, max_files=10, max_file_size_bytes=1_000_000)

    assert len(result.series) == 2


def test_discovery_file_count_limit(tmp_path: Path) -> None:
    generate_dicom_fixture_series(tmp_path, slice_count=3)

    result = discover_dicom_series(tmp_path, max_files=1, max_file_size_bytes=1_000_000)

    assert len(result.series[0].files) == 1
    assert any(skipped.reason == "file-count-limit" for skipped in result.skipped_files)


def test_fixture_overwrite_protection(tmp_path: Path) -> None:
    generate_dicom_fixture_series(tmp_path, slice_count=1)

    with pytest.raises(FileExistsError):
        generate_dicom_fixture_series(tmp_path, slice_count=1)
