"""Deterministic synthetic DICOM CT fixture generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID, CTImageStorage, ExplicitVRLittleEndian, generate_uid

from medical_imaging_platform.deidentification.policy import DIRECT_IDENTIFIER_KEYWORDS

FIXTURE_UID_ROOT = "1.2.826.0.1.3680043.10.54321.3"


def generate_dicom_fixture_series(
    output_dir: Path,
    *,
    slice_count: int = 4,
    study_uid: str | None = None,
    series_uid: str | None = None,
    malformed: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Generate a tiny deterministic CT DICOM series for tests and local validation."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    study_uid = study_uid or generate_uid(prefix=f"{FIXTURE_UID_ROOT}.1.")
    series_uid = series_uid or generate_uid(prefix=f"{FIXTURE_UID_ROOT}.2.")

    paths: list[Path] = []
    for index in range(slice_count):
        sop_uid = generate_uid(prefix=f"{FIXTURE_UID_ROOT}.3.{index + 1}.")
        dataset = build_fixture_dataset(
            index=index,
            study_uid=study_uid,
            series_uid=series_uid,
            sop_uid=sop_uid,
            malformed=malformed,
        )
        path = output_dir / f"slice-{index + 1:03d}.dcm"
        dataset.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths


def build_fixture_dataset(
    *,
    index: int,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    malformed: str | None = None,
) -> FileDataset:
    """Build one safe synthetic DICOM dataset."""
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = UID(sop_uid)
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid(prefix=f"{FIXTURE_UID_ROOT}.9.")

    dataset = FileDataset("", {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.Modality = "CT"
    dataset.BodyPartExamined = "ABDOMEN"
    dataset.StudyDate = "20260101"
    dataset.SeriesDescription = "Synthetic engineering CT fixture"
    dataset.Manufacturer = "Synthetic"
    dataset.ManufacturerModelName = "FixtureGenerator"
    dataset.PatientName = "Synthetic^Fixture"
    dataset.PatientID = "SYNTHETIC-PATIENT"
    dataset.PatientBirthDate = "19000101"
    dataset.PatientSex = "O"
    dataset.InstitutionName = "Synthetic Institution"
    dataset.AccessionNumber = "SYN-ACCESSION"
    dataset.StudyID = "SYN-STUDY"
    dataset.Rows = 8
    dataset.Columns = 8
    dataset.PixelSpacing = [2.5, 2.5]
    dataset.SliceThickness = "2.5"
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0.0, 0.0, float(index) * 2.5]
    dataset.SliceLocation = float(index) * 2.5
    dataset.InstanceNumber = index + 1
    dataset.RescaleSlope = "1"
    dataset.RescaleIntercept = "-1024"
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.BurnedInAnnotation = "NO"
    dataset.add_new((0x0011, 0x0010), "LO", "SYNTHETIC_PRIVATE_CREATOR")
    dataset.add_new((0x0011, 0x1001), "LO", "PRIVATE_SYNTHETIC_VALUE")

    pixel_array = np.full((8, 8), fill_value=index + 1, dtype=np.int16)
    dataset.PixelData = pixel_array.tobytes()

    if malformed == "wrong_modality":
        dataset.Modality = "MR"
    elif malformed == "missing_uid":
        del dataset.SeriesInstanceUID
    elif malformed == "duplicate_instance":
        dataset.InstanceNumber = 1
    elif malformed == "burned_in":
        dataset.BurnedInAnnotation = "YES"
    elif malformed == "missing_pixel_data":
        del dataset.PixelData
    elif malformed == "mixed_dimensions":
        dataset.Rows = 9
        dataset.PixelData = np.full((9, 8), fill_value=index + 1, dtype=np.int16).tobytes()

    for keyword in DIRECT_IDENTIFIER_KEYWORDS:
        if keyword not in dataset and keyword != "PatientTelephoneNumbers":
            continue
    return dataset
