"""State-dict-only checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class ClassificationCheckpointError(ValueError):
    """Raised when a classifier checkpoint is invalid."""


def save_state_dict(path: Path, model: torch.nn.Module) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save({"model_state_dict": model.state_dict()}, tmp)  # nosec B614
    tmp.replace(path)


def load_state_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ClassificationCheckpointError(f"Checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ClassificationCheckpointError("Checkpoint must contain model_state_dict.")
    state = payload["model_state_dict"]
    if not isinstance(state, dict):
        raise ClassificationCheckpointError("model_state_dict must be a mapping.")
    return state
