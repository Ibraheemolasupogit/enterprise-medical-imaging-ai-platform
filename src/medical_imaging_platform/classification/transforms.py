"""Lightweight classification transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
from monai.transforms import Compose, RandFlipd, RandGaussianNoised

from medical_imaging_platform.classification.models import ClassificationConfig


def build_transforms(config: ClassificationConfig, split: str) -> Compose:
    transforms: list[Any] = [EnsureFloatd()]
    if split == "train":
        transforms.extend(
            [
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
                RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
                EnsureFloatd(),
            ]
        )
    return Compose(transforms)


class EnsureFloatd:
    """Keep image dtype stable and labels unchanged."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)
        output["image"] = np.asarray(output["image"], dtype=np.float32)
        output["label"] = np.asarray(output["label"], dtype=np.float32)
        return output
