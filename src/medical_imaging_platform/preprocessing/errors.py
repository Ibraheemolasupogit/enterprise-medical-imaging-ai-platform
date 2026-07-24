"""Preprocessing-specific exceptions."""

from __future__ import annotations

from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class PreprocessingError(MedicalImagingPlatformError):
    """Raised when preprocessing input or configuration is invalid."""


class PreprocessingQualityError(PreprocessingError):
    """Raised when quality-control policy blocks preprocessing."""


class PreprocessingRejectedError(PreprocessingError):
    """Raised when critical or rejected data must not be preprocessed."""


class PreprocessingOutputError(PreprocessingError):
    """Raised when preprocessing output cannot be written or validated."""
