"""Safe checkpoint handling for segmentation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointError(ValueError):
    """Raised when checkpoint writing or loading fails."""


def save_state_dict(path: Path, model: torch.nn.Module) -> None:
    """Save a model state dictionary atomically where practical."""
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save({"model_state_dict": model.state_dict()}, tmp)  # nosec B614
    tmp.replace(path)


def load_state_dict(path: Path) -> dict[str, Any]:
    """Load and validate a state-dict-only checkpoint."""
    if not path.exists():
        raise CheckpointError(f"Checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise CheckpointError("Checkpoint must contain a model_state_dict mapping.")
    state = payload["model_state_dict"]
    if not isinstance(state, dict):
        raise CheckpointError("model_state_dict must be a mapping.")
    return state
