"""Validation for synthetic volumes, masks, manifests, and splits."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from medical_imaging_platform.synthetic.generator import SyntheticCase
from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class DatasetValidationError(MedicalImagingPlatformError):
    """Raised when synthetic dataset validation fails."""


def validate_case(case: SyntheticCase) -> None:
    """Validate shape, binary mask, containment, and non-empty-mask rules."""
    shape = case.previous_volume.shape
    arrays = {
        "current_volume": case.current_volume,
        "previous_body_mask": case.previous_body_mask,
        "current_body_mask": case.current_body_mask,
        "previous_left_adrenal_mask": case.previous_left_adrenal_mask,
        "previous_right_adrenal_mask": case.previous_right_adrenal_mask,
        "current_left_adrenal_mask": case.current_left_adrenal_mask,
        "current_right_adrenal_mask": case.current_right_adrenal_mask,
        "previous_lesion_mask": case.previous_lesion_mask,
        "current_lesion_mask": case.current_lesion_mask,
    }
    for name, array in arrays.items():
        if array.shape != shape:
            raise DatasetValidationError(f"{case.case_id}: {name} shape does not match volume")

    mask_arrays = {name: array for name, array in arrays.items() if name.endswith("_mask")}
    for name, mask in mask_arrays.items():
        unique_values = set(np.unique(mask.astype(np.uint8)).tolist())
        if not unique_values.issubset({0, 1}):
            raise DatasetValidationError(f"{case.case_id}: {name} is not binary")

    expected_non_empty = {
        "previous_body_mask": case.previous_body_mask,
        "current_body_mask": case.current_body_mask,
        "previous_left_adrenal_mask": case.previous_left_adrenal_mask,
        "previous_right_adrenal_mask": case.previous_right_adrenal_mask,
        "current_left_adrenal_mask": case.current_left_adrenal_mask,
        "current_right_adrenal_mask": case.current_right_adrenal_mask,
    }
    for name, mask in expected_non_empty.items():
        if not np.any(mask):
            raise DatasetValidationError(f"{case.case_id}: {name} is unexpectedly empty")

    _validate_lesion_containment(
        case.case_id,
        "previous",
        case.lesion_side,
        case.previous_lesion_mask,
        case.previous_left_adrenal_mask,
        case.previous_right_adrenal_mask,
        case.previous_body_mask,
    )
    _validate_lesion_containment(
        case.case_id,
        "current",
        case.lesion_side,
        case.current_lesion_mask,
        case.current_left_adrenal_mask,
        case.current_right_adrenal_mask,
        case.current_body_mask,
    )


def validate_cases(cases: Iterable[SyntheticCase]) -> None:
    """Validate all synthetic cases."""
    for case in cases:
        validate_case(case)


def detect_subject_leakage(
    split_map: dict[str, list[str]], subject_by_case: dict[str, str]
) -> None:
    """Ensure no research subject appears in more than one split."""
    split_by_subject: dict[str, str] = {}
    for split_name, case_ids in split_map.items():
        for case_id in case_ids:
            subject = subject_by_case[case_id]
            previous_split = split_by_subject.get(subject)
            if previous_split is not None and previous_split != split_name:
                raise DatasetValidationError(
                    f"Research subject {subject} appears in both {previous_split} and {split_name}"
                )
            split_by_subject[subject] = split_name


def _validate_lesion_containment(
    case_id: str,
    timepoint: str,
    lesion_side: str,
    lesion_mask: np.ndarray,
    left_adrenal_mask: np.ndarray,
    right_adrenal_mask: np.ndarray,
    body_mask: np.ndarray,
) -> None:
    if not np.any(lesion_mask):
        return
    roi_mask = left_adrenal_mask if lesion_side == "left" else right_adrenal_mask
    if np.any(lesion_mask & ~roi_mask):
        raise DatasetValidationError(f"{case_id}: {timepoint} lesion is outside configured ROI")
    if np.any(lesion_mask & ~body_mask):
        raise DatasetValidationError(f"{case_id}: {timepoint} lesion overlaps invalid background")
