"""Shared segmentation pipeline utilities."""

from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path

import monai
import numpy as np
import torch

from medical_imaging_platform.segmentation.models import SegmentationConfig
from medical_imaging_platform.synthetic.manifest import sha256_file


def set_reproducibility(seed: int) -> None:
    """Set Python, NumPy, PyTorch, and MONAI deterministic seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    monai.utils.set_determinism(seed=seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device, keeping CPU as the safe default."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but not available.")
    return torch.device(requested)


def configuration_checksum(config: SegmentationConfig) -> str:
    """Checksum stable configuration serialisation."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def dependency_versions() -> dict[str, str]:
    """Return dependency versions for experiment evidence."""
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "monai_version": str(monai.__version__),
    }


def safe_experiment_dir(output_root: Path, experiment_id: str) -> Path:
    """Build an experiment directory with path traversal protection."""
    root = output_root.resolve()
    output_dir = (root / experiment_id).resolve()
    if os.path.commonpath([str(root), str(output_dir)]) != str(root):
        raise ValueError("Experiment output path escapes configured root.")
    return output_dir


def manifest_checksum(path: Path) -> str:
    """Checksum a dataset or model manifest file."""
    return sha256_file(path)
