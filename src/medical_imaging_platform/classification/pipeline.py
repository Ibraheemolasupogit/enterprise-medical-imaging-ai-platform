"""Shared classification reproducibility and path helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from pathlib import Path

import monai
import numpy as np
import torch

from medical_imaging_platform.classification.models import ClassificationConfig
from medical_imaging_platform.synthetic.manifest import sha256_file


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    monai.utils.set_determinism(seed=seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but not available.")
    return torch.device(requested)


def configuration_checksum(config: ClassificationConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dependency_versions() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "monai_version": str(monai.__version__),
    }


def safe_output_dir(root: Path, child: str) -> Path:
    resolved_root = root.resolve()
    output_dir = (resolved_root / child).resolve()
    if os.path.commonpath([str(resolved_root), str(output_dir)]) != str(resolved_root):
        raise ValueError("Output path escapes configured root.")
    return output_dir


def manifest_checksum(path: Path) -> str:
    return sha256_file(path)
