"""Safe state-dict-only checkpoint handling for detection experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


class DetectionCheckpointError(MedicalImagingPlatformError):
    """Raised when a detection checkpoint cannot be read."""


def save_state_dict(path: Path, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)


def load_state_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DetectionCheckpointError(f"Detection checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise DetectionCheckpointError("Checkpoint must contain a model_state_dict mapping.")
    state = payload["model_state_dict"]
    if not isinstance(state, dict):
        raise DetectionCheckpointError("model_state_dict must be a mapping.")
    return state
