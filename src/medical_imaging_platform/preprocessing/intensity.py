"""Intensity clipping/windowing and normalisation."""

from __future__ import annotations

import numpy as np

from medical_imaging_platform.preprocessing.models import (
    IntensityTransformSummary,
    PreprocessingConfig,
)


def apply_intensity_transform(
    volume: np.ndarray,
    *,
    config: PreprocessingConfig,
    profile_name: str | None = None,
) -> tuple[np.ndarray, IntensityTransformSummary, list[str]]:
    """Apply deterministic technical intensity transforms."""
    warnings: list[str] = []
    selected_profile = profile_name or config.default_intensity_profile
    if selected_profile not in config.intensity_profiles:
        raise ValueError(f"Unknown intensity profile: {selected_profile}")
    profile = config.intensity_profiles[selected_profile]

    input_min = float(np.min(volume))
    input_max = float(np.max(volume))
    clipped = volume.astype(config.pixel_dtype, copy=True)
    clipped_voxels = 0
    if profile.lower is not None:
        clipped_voxels += int(np.count_nonzero(clipped < profile.lower))
    if profile.upper is not None:
        clipped_voxels += int(np.count_nonzero(clipped > profile.upper))
    if profile.lower is not None or profile.upper is not None:
        clipped = np.clip(clipped, profile.lower, profile.upper)

    clipped_min = float(np.min(clipped))
    clipped_max = float(np.max(clipped))
    mean = float(np.mean(clipped))
    std = float(np.std(clipped))
    fallback_used = False

    if config.normalisation_mode == "minmax":
        denominator = clipped_max - clipped_min
        if denominator == 0:
            transformed = np.zeros_like(clipped, dtype=config.pixel_dtype)
            fallback_used = True
            warnings.append("Min-max normalisation encountered a constant volume; output is zeros.")
        else:
            transformed = ((clipped - clipped_min) / denominator).astype(config.pixel_dtype)
    elif config.normalisation_mode == "zscore":
        if std == 0:
            transformed = np.zeros_like(clipped, dtype=config.pixel_dtype)
            fallback_used = True
            warnings.append(
                "Z-score normalisation encountered zero standard deviation; output is zeros."
            )
        else:
            transformed = ((clipped - mean) / std).astype(config.pixel_dtype)
    else:
        transformed = clipped.astype(config.pixel_dtype)

    summary = IntensityTransformSummary(
        profile_name=selected_profile,
        clipping_lower=profile.lower,
        clipping_upper=profile.upper,
        normalisation_mode=config.normalisation_mode,
        input_range=(input_min, input_max),
        clipped_range=(clipped_min, clipped_max),
        output_range=(float(np.min(transformed)), float(np.max(transformed))),
        clipped_voxel_count=clipped_voxels,
        mean_before_normalisation=mean,
        std_before_normalisation=std,
        fallback_used=fallback_used,
        output_dtype=str(transformed.dtype),
    )
    return transformed, summary, warnings
