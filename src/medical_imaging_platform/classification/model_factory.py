"""Small 3D lesion-presence classifier."""

from __future__ import annotations

import torch

from medical_imaging_platform.classification.models import ClassificationConfig


class Small3DClassifier(torch.nn.Module):  # type: ignore[misc]
    """Configurable compact 3D CNN with global average pooling."""

    def __init__(self, channels: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.extend(
                [
                    torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                    torch.nn.InstanceNorm3d(out_channels),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool3d(kernel_size=2),
                ]
            )
            in_channels = out_channels
        self.features = torch.nn.Sequential(*layers)
        self.pool = torch.nn.AdaptiveAvgPool3d(1)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(channels[-1], 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        pooled = self.pool(features).flatten(1)
        return self.head(self.dropout(pooled)).squeeze(1)


def build_classifier(config: ClassificationConfig) -> torch.nn.Module:
    """Build deterministic small 3D classifier."""
    torch.manual_seed(config.random_seed)
    return Small3DClassifier(config.channels, config.dropout)


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
