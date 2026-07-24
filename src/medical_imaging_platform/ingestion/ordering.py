"""Deterministic DICOM slice ordering."""

from __future__ import annotations

import math

import numpy as np

from medical_imaging_platform.ingestion.models import (
    DicomFileMetadata,
    OrderedSliceSet,
    OrderingIssue,
)


def order_slices(metadata: list[DicomFileMetadata]) -> OrderedSliceSet:
    """Order slices using position, slice location, instance number, then filename."""
    issues = _common_ordering_issues(metadata)
    if not metadata:
        return OrderedSliceSet(strategy="filename", files=[], issues=issues)

    if all(item.image_orientation_patient and item.image_position_patient for item in metadata):
        orientation = metadata[0].image_orientation_patient
        if orientation and all(item.image_orientation_patient == orientation for item in metadata):
            normal = _slice_normal(orientation)
            positions = [
                float(np.dot(np.array(item.image_position_patient), normal))
                for item in metadata
                if item.image_position_patient
            ]
            return _ordered_by_values(metadata, positions, "image_position_patient", issues)
        issues.append(OrderingIssue(severity="ERROR", message="Inconsistent orientation metadata"))

    if all(item.slice_location is not None for item in metadata):
        return _ordered_by_values(
            metadata,
            [float(item.slice_location) for item in metadata if item.slice_location is not None],
            "slice_location",
            issues,
        )

    if all(item.instance_number is not None for item in metadata):
        return _ordered_by_values(
            metadata,
            [float(item.instance_number) for item in metadata if item.instance_number is not None],
            "instance_number",
            issues,
        )

    issues.append(
        OrderingIssue(
            severity="WARNING", message="Missing ordering metadata; using filename fallback"
        )
    )
    return OrderedSliceSet(
        strategy="filename", files=sorted(metadata, key=lambda item: item.file_path), issues=issues
    )


def _ordered_by_values(
    metadata: list[DicomFileMetadata],
    values: list[float],
    strategy: str,
    issues: list[OrderingIssue],
) -> OrderedSliceSet:
    rounded = [round(value, 4) for value in values]
    if len(set(rounded)) != len(rounded):
        issues.append(
            OrderingIssue(severity="ERROR", message=f"Duplicate positions for {strategy}")
        )
    if len(values) > 1:
        diffs = np.diff(values)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            issues.append(
                OrderingIssue(severity="WARNING", message=f"Non-monotonic positions for {strategy}")
            )
    ordered = [
        item
        for _, item in sorted(
            zip(values, metadata, strict=True), key=lambda pair: (pair[0], pair[1].file_path)
        )
    ]
    return OrderedSliceSet(strategy=strategy, files=ordered, issues=issues)  # type: ignore[arg-type]


def _common_ordering_issues(metadata: list[DicomFileMetadata]) -> list[OrderingIssue]:
    issues: list[OrderingIssue] = []
    instances = [item.instance_number for item in metadata if item.instance_number is not None]
    if len(set(instances)) != len(instances):
        issues.append(OrderingIssue(severity="ERROR", message="Duplicate instance numbers"))
    if any(
        item.image_position_patient is None
        and item.slice_location is None
        and item.instance_number is None
        for item in metadata
    ):
        issues.append(
            OrderingIssue(
                severity="WARNING", message="One or more slices missing ordering metadata"
            )
        )
    return issues


def _slice_normal(orientation: tuple[float, float, float, float, float, float]) -> np.ndarray:
    row = np.array(orientation[:3], dtype=float)
    column = np.array(orientation[3:], dtype=float)
    normal = np.cross(row, column)
    norm = np.linalg.norm(normal)
    if math.isclose(norm, 0.0):
        return np.array([0.0, 0.0, 1.0])
    return np.asarray(normal / norm, dtype=float)
