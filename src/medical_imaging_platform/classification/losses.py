"""Classification losses."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from medical_imaging_platform.classification.models import ClassificationConfig


def classification_loss(
    logits: torch.Tensor, target: torch.Tensor, config: ClassificationConfig, positive_weight: float
) -> torch.Tensor:
    """Binary cross-entropy with optional positive-class weighting."""
    weight = torch.as_tensor(positive_weight, dtype=logits.dtype, device=logits.device)
    return functional.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
