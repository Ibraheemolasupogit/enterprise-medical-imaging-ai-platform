# mypy: disable-error-code="misc"
"""Classification model factory."""

from __future__ import annotations

from typing import Any

import torch

from medical_imaging_platform.classification.models import ClassificationConfig


class Small3DClassifier(torch.nn.Module):
    """Configurable compact 3D CNN with global average pooling."""

    def __init__(self, channels: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.extend(
                [
                    torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                    torch.nn.BatchNorm3d(out_channels),
                    torch.nn.GELU(),
                    torch.nn.MaxPool3d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels
        self.features = torch.nn.Sequential(*layers)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(channels[-1], 1)

    @property
    def feature_dim(self) -> int:
        return int(self.head.in_features)

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        return torch.nn.functional.adaptive_avg_pool3d(features, output_size=1).flatten(1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.extract_features(image))).squeeze(1)


class Tiny3DViTClassifier(torch.nn.Module):
    """Small native 3D ViT for deterministic engineering benchmarks."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        embedding_dim: int,
        transformer_depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.grid_size = tuple(
            axis // patch for axis, patch in zip(input_shape, patch_size, strict=True)
        )
        token_count = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.patch_embed = torch.nn.Conv3d(
            1, embedding_dim, kernel_size=patch_size, stride=patch_size
        )
        self.class_token = torch.nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = torch.nn.Parameter(torch.zeros(1, token_count + 1, embedding_dim))
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=int(embedding_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        self.norm = torch.nn.LayerNorm(embedding_dim)
        self.head = torch.nn.Linear(embedding_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch.nn.init.trunc_normal_(self.position_embedding, std=0.02)
        torch.nn.init.trunc_normal_(self.class_token, std=0.02)

    @property
    def feature_dim(self) -> int:
        return int(self.head.in_features)

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(image).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((class_token, tokens), dim=1) + self.position_embedding)
        return self.norm(encoded[:, 0])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_features(image)).squeeze(1)


def build_classifier(config: ClassificationConfig) -> torch.nn.Module:
    """Build a deterministic classifier for the configured model family."""
    torch.manual_seed(config.random_seed)
    if config.model_family == "small_3d_cnn":
        return Small3DClassifier(config.channels, config.dropout)
    if config.model_family == "tiny_3d_vit":
        return Tiny3DViTClassifier(
            input_shape=config.input_shape,
            patch_size=config.patch_size,
            embedding_dim=config.embedding_dim,
            transformer_depth=config.transformer_depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.transformer_dropout,
        )
    raise ValueError(f"Unsupported model family: {config.model_family}")


def architecture_metadata(config: ClassificationConfig) -> dict[str, Any]:
    if config.model_family == "small_3d_cnn":
        return {
            "family": "small_3d_cnn",
            "channels": list(config.channels),
            "dropout": config.dropout,
            "embedding_dim": config.channels[-1],
            "output_logits": 1,
        }
    patch_grid = [
        axis // patch for axis, patch in zip(config.input_shape, config.patch_size, strict=True)
    ]
    return {
        "family": "tiny_3d_vit",
        "patch_size": list(config.patch_size),
        "patch_grid": patch_grid,
        "token_count": int(patch_grid[0] * patch_grid[1] * patch_grid[2] + 1),
        "embedding_dim": config.embedding_dim,
        "transformer_depth": config.transformer_depth,
        "num_heads": config.num_heads,
        "mlp_ratio": config.mlp_ratio,
        "transformer_dropout": config.transformer_dropout,
        "output_logits": 1,
    }


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
