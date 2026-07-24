"""NumPy [z, y, x] to SimpleITK [x, y, z] conversion utilities."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk  # noqa: N813

from medical_imaging_platform.preprocessing.models import PreprocessingResult


def numpy_to_sitk(volume_zyx: np.ndarray, metadata: PreprocessingResult) -> sitk.Image:
    """Convert internal [z, y, x] NumPy volume to a SimpleITK image."""
    if volume_zyx.ndim != 3:
        raise ValueError("registration volumes must be 3D [z, y, x] arrays")
    image = sitk.GetImageFromArray(volume_zyx.astype(np.float32))
    spacing_zyx = metadata.spacing_mm
    image.SetSpacing((float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0])))
    image.SetOrigin(_origin_xyz(metadata))
    direction = _direction_xyz(metadata)
    if direction is not None:
        image.SetDirection(direction)
    return image


def sitk_to_numpy(image: sitk.Image) -> np.ndarray:
    """Convert a SimpleITK image back to internal [z, y, x] NumPy order."""
    return np.asarray(sitk.GetArrayFromImage(image), dtype=np.float32)


def validate_round_trip(volume_zyx: np.ndarray, metadata: PreprocessingResult) -> None:
    """Validate shape and spacing mapping across conversion boundaries."""
    image = numpy_to_sitk(volume_zyx, metadata)
    round_trip = sitk_to_numpy(image)
    if tuple(round_trip.shape) != tuple(volume_zyx.shape):
        raise ValueError("NumPy/SimpleITK round-trip shape mismatch")
    expected_spacing = (metadata.spacing_mm[2], metadata.spacing_mm[1], metadata.spacing_mm[0])
    if tuple(float(item) for item in image.GetSpacing()) != expected_spacing:
        raise ValueError("NumPy/SimpleITK spacing mapping mismatch")


def _origin_xyz(metadata: PreprocessingResult) -> tuple[float, float, float]:
    positions = metadata.geometry.image_positions_patient
    first = positions[0] if positions else None
    if first is None:
        return (0.0, 0.0, 0.0)
    return (float(first[0]), float(first[1]), float(first[2]))


def _direction_xyz(metadata: PreprocessingResult) -> tuple[float, ...] | None:
    row = metadata.geometry.row_direction_cosines
    column = metadata.geometry.column_direction_cosines
    normal = metadata.geometry.slice_normal
    if row is None or column is None or normal is None:
        return None
    matrix = np.array([row, column, normal], dtype=float).T
    return tuple(float(item) for item in matrix.reshape(-1))
