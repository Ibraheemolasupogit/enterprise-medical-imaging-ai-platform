"""Deterministic engineering crops and padding."""

from __future__ import annotations

import numpy as np

from medical_imaging_platform.preprocessing.errors import PreprocessingError
from medical_imaging_platform.preprocessing.models import CropPaddingSummary, PreprocessingConfig


def apply_crop_and_padding(
    volume: np.ndarray, *, config: PreprocessingConfig
) -> tuple[np.ndarray, CropPaddingSummary, list[str]]:
    """Apply non-anatomical crop/pad transforms in [z, y, x] order."""
    warnings: list[str] = []
    input_shape = _shape3(volume.shape)
    bounds = _crop_bounds(volume, config)
    cropped = volume[
        bounds["z"][0] : bounds["z"][1],
        bounds["y"][0] : bounds["y"][1],
        bounds["x"][0] : bounds["x"][1],
    ]
    if cropped.size == 0:
        raise PreprocessingError("Crop produced an empty volume.")

    cropped_shape = _shape3(cropped.shape)
    pad_widths = _pad_widths(cropped_shape, config.minimum_output_shape)
    if any(before or after for before, after in pad_widths):
        warnings.append("Minimum output shape required deterministic padding.")
        cropped = np.pad(
            cropped,
            pad_widths,
            mode="constant",
            constant_values=float(config.padding_value),
        )

    summary = CropPaddingSummary(
        crop_mode=config.crop_mode,
        input_shape=input_shape,
        crop_bounds_zyx=bounds,
        crop_offsets_zyx=(bounds["z"][0], bounds["y"][0], bounds["x"][0]),
        cropped_shape=cropped_shape,
        padding_value=float(config.padding_value),
        pad_widths_zyx=pad_widths,
        output_shape=_shape3(cropped.shape),
    )
    return cropped, summary, warnings


def _crop_bounds(volume: np.ndarray, config: PreprocessingConfig) -> dict[str, tuple[int, int]]:
    shape = _shape3(volume.shape)
    if config.crop_mode == "none":
        return _full_bounds(shape)
    if config.crop_mode == "centre":
        assert config.centre_crop_shape is not None
        return _centre_bounds(shape, config.centre_crop_shape)
    if config.crop_mode == "fixed":
        assert config.fixed_crop_bounds is not None
        bounds = {
            "z": config.fixed_crop_bounds.z,
            "y": config.fixed_crop_bounds.y,
            "x": config.fixed_crop_bounds.x,
        }
        _validate_bounds(bounds, shape)
        return bounds
    return _non_background_bounds(volume, config.background_threshold)


def _full_bounds(shape: tuple[int, int, int]) -> dict[str, tuple[int, int]]:
    return {"z": (0, shape[0]), "y": (0, shape[1]), "x": (0, shape[2])}


def _centre_bounds(
    shape: tuple[int, int, int], target: tuple[int, int, int]
) -> dict[str, tuple[int, int]]:
    starts = [
        max((size - requested) // 2, 0) for size, requested in zip(shape, target, strict=True)
    ]
    ends = [
        min(start + requested, size)
        for start, requested, size in zip(starts, target, shape, strict=True)
    ]
    return {"z": (starts[0], ends[0]), "y": (starts[1], ends[1]), "x": (starts[2], ends[2])}


def _non_background_bounds(volume: np.ndarray, threshold: float) -> dict[str, tuple[int, int]]:
    mask = np.abs(volume) > threshold
    if not np.any(mask):
        raise PreprocessingError("Non-background crop found no foreground voxels.")
    indices = np.argwhere(mask)
    lower = indices.min(axis=0)
    upper = indices.max(axis=0) + 1
    return {
        "z": (int(lower[0]), int(upper[0])),
        "y": (int(lower[1]), int(upper[1])),
        "x": (int(lower[2]), int(upper[2])),
    }


def _validate_bounds(bounds: dict[str, tuple[int, int]], shape: tuple[int, int, int]) -> None:
    for axis, size in zip(("z", "y", "x"), shape, strict=True):
        start, end = bounds[axis]
        if start < 0 or end > size or start >= end:
            raise PreprocessingError(f"Fixed crop bounds for {axis} are outside volume shape.")


def _pad_widths(
    shape: tuple[int, ...], minimum_shape: tuple[int, int, int] | None
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if minimum_shape is None:
        return ((0, 0), (0, 0), (0, 0))
    pads: list[tuple[int, int]] = []
    for size, minimum in zip(shape, minimum_shape, strict=True):
        missing = max(minimum - int(size), 0)
        before = missing // 2
        after = missing - before
        pads.append((before, after))
    return (pads[0], pads[1], pads[2])


def _shape3(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise PreprocessingError("Preprocessing volumes must be 3D [z, y, x] arrays.")
    return (int(shape[0]), int(shape[1]), int(shape[2]))
