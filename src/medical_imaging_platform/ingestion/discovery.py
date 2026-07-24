"""Recursive DICOM series discovery."""

from __future__ import annotations

from pathlib import Path

from medical_imaging_platform.ingestion.loader import DicomLoadError, load_dicom
from medical_imaging_platform.ingestion.metadata import extract_metadata
from medical_imaging_platform.ingestion.models import (
    DicomDiscoveryResult,
    DicomSeries,
    SkippedDicomFile,
)


def discover_dicom_series(
    input_dir: Path,
    *,
    max_files: int,
    max_file_size_bytes: int,
) -> DicomDiscoveryResult:
    """Discover DICOM files recursively and group them by study and series UID."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise DicomLoadError(f"Input directory does not exist: {input_dir}")

    candidates = sorted(
        path for path in input_dir.rglob("*") if path.is_file() and not path.is_symlink()
    )
    skipped: list[SkippedDicomFile] = []
    grouped: dict[tuple[str, str], list[str]] = {}
    for index, path in enumerate(candidates):
        if index >= max_files:
            skipped.append(SkippedDicomFile(path=str(path), reason="file-count-limit"))
            continue
        try:
            dataset = load_dicom(path, header_only=True, max_file_size_bytes=max_file_size_bytes)
            metadata = extract_metadata(dataset, path)
        except DicomLoadError as exc:
            skipped.append(SkippedDicomFile(path=str(path), reason=str(exc)))
            continue
        if metadata.study_instance_uid is None or metadata.series_instance_uid is None:
            skipped.append(SkippedDicomFile(path=str(path), reason="missing-study-or-series-uid"))
            continue
        grouped.setdefault((metadata.study_instance_uid, metadata.series_instance_uid), []).append(
            str(path)
        )

    series = [
        DicomSeries(
            study_instance_uid=study_uid, series_instance_uid=series_uid, files=sorted(files)
        )
        for (study_uid, series_uid), files in sorted(grouped.items())
    ]
    return DicomDiscoveryResult(series=series, skipped_files=skipped)
