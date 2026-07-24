"""Safe pydicom loading helpers."""

from __future__ import annotations

from pathlib import Path

import pydicom
from pydicom.dataset import FileDataset

from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class DicomLoadError(MedicalImagingPlatformError):
    """Raised when a DICOM file cannot be safely loaded."""


def load_dicom(path: Path, *, header_only: bool = True, max_file_size_bytes: int) -> FileDataset:
    """Load a DICOM file with size checks and optional pixel skipping."""
    if path.is_symlink():
        raise DicomLoadError(f"Refusing to follow symbolic link: {path}")
    if not path.exists() or not path.is_file():
        raise DicomLoadError(f"DICOM path is not a file: {path}")
    size = path.stat().st_size
    if size > max_file_size_bytes:
        raise DicomLoadError(f"DICOM file exceeds maximum size: {path}")
    try:
        return pydicom.dcmread(path, stop_before_pixels=header_only, force=False)
    except Exception as exc:  # pydicom raises several parsing exception types.
        raise DicomLoadError(f"Unable to read DICOM file {path}: {exc}") from exc
