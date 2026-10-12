"""Shared deterministic helpers for detection experiments."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from medical_imaging_platform.detection.models import DetectionConfig
from medical_imaging_platform.synthetic.manifest import sha256_file


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but is not available.")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    return torch.device(device_name)


def safe_output_dir(root: Path, child: str) -> Path:
    resolved_root = root.resolve()
    child_path = (resolved_root / child).resolve()
    import os

    if os.path.commonpath([str(resolved_root), str(child_path)]) != str(resolved_root):
        raise ValueError("Detection output path escapes configured root.")
    return child_path


def configuration_checksum(config: DetectionConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_checksum(path: Path) -> str:
    return sha256_file(path)


def dependency_versions() -> dict[str, str]:
    versions = {
        "python_version": ".".join(str(part) for part in __import__("sys").version_info[:3]),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    try:
        import torchvision  # type: ignore[import-untyped]

        versions["torchvision_version"] = torchvision.__version__
    except Exception:
        versions["torchvision_version"] = "not_available"
    return versions


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
