"""Segmentation losses for synthetic binary masks."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from medical_imaging_platform.segmentation.models import SegmentationConfig


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss with stable empty-mask handling."""
    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * target)
    denominator = torch.sum(probs) + torch.sum(target)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice


def segmentation_loss(
    logits: torch.Tensor, target: torch.Tensor, config: SegmentationConfig
) -> torch.Tensor:
    """Compute configured Dice, BCE, or combined Dice/BCE loss."""
    total = torch.zeros((), dtype=logits.dtype, device=logits.device)
    if config.loss in {"dice", "dice_bce"} and config.dice_weight > 0:
        total = total + config.dice_weight * dice_loss(logits, target)
    if config.loss in {"bce", "dice_bce"} and config.bce_weight > 0:
        total = total + config.bce_weight * functional.binary_cross_entropy_with_logits(
            logits, target
        )
    return total
