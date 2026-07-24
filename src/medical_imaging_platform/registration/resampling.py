"""Registration resampling utilities."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk  # noqa: N813

from medical_imaging_platform.registration.conversion import sitk_to_numpy


def resample_moving_to_fixed(
    moving_image: sitk.Image,
    fixed_image: sitk.Image,
    transform: sitk.Transform,
    *,
    interpolator: str,
    default_pixel_value: float,
) -> np.ndarray:
    """Apply the registration transform with fixed-image output geometry."""
    sitk_interpolator = sitk.sitkNearestNeighbor if interpolator == "nearest" else sitk.sitkLinear
    resampled = sitk.Resample(
        moving_image,
        fixed_image,
        transform,
        sitk_interpolator,
        float(default_pixel_value),
        moving_image.GetPixelID(),
    )
    array = sitk_to_numpy(resampled)
    if not np.all(np.isfinite(array)):
        raise ValueError("Registered moving volume contains non-finite values.")
    return array
