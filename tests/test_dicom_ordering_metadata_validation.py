from pathlib import Path

import pydicom
from pydicom.uid import UID

from medical_imaging_platform.ingestion.fixtures import generate_dicom_fixture_series
from medical_imaging_platform.ingestion.loader import load_dicom
from medical_imaging_platform.ingestion.metadata import (
    detect_identifying_keywords,
    extract_metadata,
)
from medical_imaging_platform.ingestion.ordering import order_slices
from medical_imaging_platform.ingestion.validation import validate_series


def _metadata_for(paths: list[Path]):
    return [
        extract_metadata(load_dicom(path, header_only=True, max_file_size_bytes=1_000_000), path)
        for path in paths
    ]


def test_image_position_patient_ordering(tmp_path: Path) -> None:
    paths = list(reversed(generate_dicom_fixture_series(tmp_path, slice_count=3)))

    ordered = order_slices(_metadata_for(paths))

    assert ordered.strategy == "image_position_patient"
    assert [item.instance_number for item in ordered.files] == [1, 2, 3]


def test_slice_location_fallback(tmp_path: Path) -> None:
    paths = generate_dicom_fixture_series(tmp_path, slice_count=2)
    for path in paths:
        dataset = pydicom.dcmread(path)
        del dataset.ImagePositionPatient
        dataset.save_as(path)

    ordered = order_slices(_metadata_for(paths))

    assert ordered.strategy == "slice_location"


def test_instance_number_fallback(tmp_path: Path) -> None:
    paths = generate_dicom_fixture_series(tmp_path, slice_count=2)
    for path in paths:
        dataset = pydicom.dcmread(path)
        del dataset.ImagePositionPatient
        del dataset.SliceLocation
        dataset.save_as(path)

    ordered = order_slices(_metadata_for(paths))

    assert ordered.strategy == "instance_number"


def test_duplicate_positions_and_instances(tmp_path: Path) -> None:
    paths = generate_dicom_fixture_series(tmp_path, slice_count=2)
    dataset = pydicom.dcmread(paths[1])
    dataset.ImagePositionPatient = [0.0, 0.0, 0.0]
    dataset.InstanceNumber = 1
    dataset.save_as(paths[1])

    ordered = order_slices(_metadata_for(paths))

    assert any("Duplicate positions" in issue.message for issue in ordered.issues)
    assert any("Duplicate instance" in issue.message for issue in ordered.issues)


def test_missing_ordering_fields_filename_fallback(tmp_path: Path) -> None:
    paths = generate_dicom_fixture_series(tmp_path, slice_count=2)
    for path in paths:
        dataset = pydicom.dcmread(path)
        del dataset.ImagePositionPatient
        del dataset.SliceLocation
        del dataset.InstanceNumber
        dataset.save_as(path)

    ordered = order_slices(_metadata_for(paths))

    assert ordered.strategy == "filename"
    assert any("filename fallback" in issue.message for issue in ordered.issues)


def test_inconsistent_orientation(tmp_path: Path) -> None:
    paths = generate_dicom_fixture_series(tmp_path, slice_count=2)
    dataset = pydicom.dcmread(paths[1])
    dataset.ImageOrientationPatient = [0, 1, 0, 1, 0, 0]
    dataset.save_as(paths[1])

    ordered = order_slices(_metadata_for(paths))

    assert any("Inconsistent orientation" in issue.message for issue in ordered.issues)


def test_metadata_extraction_and_no_identifier_values_in_safe_output(tmp_path: Path) -> None:
    path = generate_dicom_fixture_series(tmp_path, slice_count=1)[0]
    dataset = load_dicom(path, header_only=True, max_file_size_bytes=1_000_000)
    metadata = extract_metadata(dataset, path)

    assert metadata.modality == "CT"
    assert metadata.pixel_spacing == (2.5, 2.5)
    assert "Synthetic^Fixture" not in metadata.model_dump_json()
    assert detect_identifying_keywords(dataset)


def test_validation_valid_ct_series(tmp_path: Path) -> None:
    paths = generate_dicom_fixture_series(tmp_path, slice_count=2)

    findings = validate_series(
        paths, accepted_modality="CT", max_file_size_bytes=1_000_000, require_pixel_data=True
    )

    assert not any(finding.severity == "ERROR" for finding in findings)


def test_validation_wrong_modality_duplicate_sop_and_mixed_dimensions(tmp_path: Path) -> None:
    paths = generate_dicom_fixture_series(tmp_path, slice_count=2)
    first = pydicom.dcmread(paths[0])
    second = pydicom.dcmread(paths[1])
    second.Modality = "MR"
    second.SOPInstanceUID = first.SOPInstanceUID
    second.Rows = 9
    second.save_as(paths[1])

    findings = validate_series(paths, accepted_modality="CT", max_file_size_bytes=1_000_000)
    messages = " ".join(finding.message for finding in findings)

    assert "Expected modality CT" in messages
    assert "Duplicate SOP" in messages
    assert "rows is inconsistent" in messages


def test_validation_missing_pixel_data_and_unsupported_transfer_syntax(tmp_path: Path) -> None:
    path = generate_dicom_fixture_series(tmp_path, slice_count=1)[0]
    dataset = pydicom.dcmread(path)
    del dataset.PixelData
    dataset.file_meta.TransferSyntaxUID = UID("1.2.840.10008.1.2.4.90")
    dataset.save_as(path)

    findings = validate_series(
        [path], accepted_modality="CT", max_file_size_bytes=1_000_000, require_pixel_data=True
    )
    rule_ids = {finding.rule_id for finding in findings}

    assert "PIXEL_DATA_PRESENT" in rule_ids
    assert "TRANSFER_SYNTAX" in rule_ids
