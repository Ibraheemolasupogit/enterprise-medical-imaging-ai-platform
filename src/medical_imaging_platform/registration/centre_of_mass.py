"""Centre-of-mass translation baseline."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk  # noqa: N813

from medical_imaging_platform.registration.metrics import mask_centre_of_mass


def foreground_mask(volume: np.ndarray, threshold: float) -> np.ndarray:
    """Build an engineering foreground mask from an intensity threshold."""
    return np.asarray(np.abs(volume) > threshold)


def centre_of_mass_transform(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    spacing_zyx: tuple[float, float, float],
    threshold: float,
) -> tuple[sitk.TranslationTransform, tuple[float, float, float], tuple[float, float, float]]:
    """Return a SimpleITK translation transform from moving COM to fixed COM."""
    fixed_com = mask_centre_of_mass(foreground_mask(fixed, threshold), spacing_zyx)
    moving_com = mask_centre_of_mass(foreground_mask(moving, threshold), spacing_zyx)
    if fixed_com is None or moving_com is None:
        raise ValueError("Foreground could not be established for centre-of-mass registration.")
    translation_xyz = tuple(float(moving_com[index] - fixed_com[index]) for index in range(3))
    transform = sitk.TranslationTransform(3)
    transform.SetOffset(translation_xyz)
    translation_voxels_zyx = (
        -translation_xyz[2] / spacing_zyx[0],
        -translation_xyz[1] / spacing_zyx[1],
        -translation_xyz[0] / spacing_zyx[2],
    )
    moving_to_fixed_xyz = (
        -float(translation_xyz[0]),
        -float(translation_xyz[1]),
        -float(translation_xyz[2]),
    )
    return transform, moving_to_fixed_xyz, translation_voxels_zyx
