"""Small MONAI 3D U-Net construction."""

from __future__ import annotations

import torch
from monai.networks.nets import UNet

from medical_imaging_platform.segmentation.models import SegmentationConfig


def build_unet(config: SegmentationConfig) -> torch.nn.Module:
    """Build the configured small 3D U-Net baseline."""
    torch.manual_seed(config.random_seed)
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=config.channels,
        strides=config.strides,
        num_res_units=config.num_res_units,
    )


def parameter_count(model: torch.nn.Module) -> int:
    """Return trainable and non-trainable parameter count."""
    return int(sum(parameter.numel() for parameter in model.parameters()))
