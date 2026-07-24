"""Deterministic MONAI-compatible transforms for segmentation."""

from __future__ import annotations

from typing import Any

import numpy as np
from monai.transforms import Compose, RandFlipd, RandGaussianNoised

from medical_imaging_platform.segmentation.models import SegmentationConfig


def build_transforms(config: SegmentationConfig, split: str) -> Compose:
    """Build lightweight transforms with aligned image/mask spatial operations."""
    transforms: list[Any] = [EnsureBinaryMaskd()]
    if split == "train" and config.augmentation.enabled:
        transforms.extend(
            [
                RandFlipd(
                    keys=["image", "mask"],
                    prob=config.augmentation.flip_probability,
                    spatial_axis=0,
                ),
                RandGaussianNoised(
                    keys=["image"],
                    prob=config.augmentation.flip_probability,
                    mean=0.0,
                    std=config.augmentation.noise_std,
                ),
                EnsureBinaryMaskd(),
            ]
        )
    return Compose(transforms)


class EnsureBinaryMaskd:
    """Keep masks binary after any stochastic transform."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)
        output["image"] = np.asarray(output["image"], dtype=np.float32)
        output["mask"] = (np.asarray(output["mask"]) > 0.5).astype(np.float32)
        return output
