"""Local filesystem security controls for the API."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from medical_imaging_platform.api.errors import APIError
from medical_imaging_platform.api.models import APIConfig, ArrayPayload


def resolve_allowed_path(path_text: str, roots: list[Path], *, must_exist: bool = True) -> Path:
    """Resolve a local path and ensure it stays under configured roots."""
    if "://" in path_text:
        raise APIError("API-AUTH-403", "Remote paths are not allowed.", status_code=403)
    path = Path(path_text)
    if path.suffix != ".npy":
        raise APIError("API-VALID-422", "Only .npy arrays are supported.", status_code=422)
    if path.is_symlink():
        raise APIError("API-AUTH-403", "Symlink inputs are not allowed.", status_code=403)
    resolved = path.resolve()
    if must_exist and not resolved.exists():
        raise APIError("API-NOTFOUND-404", "Input evidence was not found.", status_code=404)
    root_paths = [root.resolve() for root in roots]
    if not any(os.path.commonpath([str(root), str(resolved)]) == str(root) for root in root_paths):
        raise APIError("API-AUTH-403", "Path is outside configured API roots.", status_code=403)
    return resolved


def load_npy(path: Path, config: APIConfig) -> np.ndarray:
    if path.stat().st_size > config.maximum_array_bytes:
        raise APIError("API-SIZE-413", "Array file exceeds configured size limit.", status_code=413)
    try:
        array = np.load(path, allow_pickle=False)
    except ValueError as exc:
        raise APIError("API-VALID-422", "Invalid NumPy array.", status_code=422) from exc
    return validate_array(array, config)


def array_from_payload(payload: ArrayPayload, config: APIConfig) -> np.ndarray:
    if len(payload.values) != int(np.prod(payload.shape)):
        raise APIError(
            "API-VALID-422", "Array payload length does not match shape.", status_code=422
        )
    if len(payload.values) * 8 > config.maximum_array_bytes:
        raise APIError(
            "API-SIZE-413", "Array payload exceeds configured size limit.", status_code=413
        )
    return validate_array(
        np.asarray(payload.values, dtype=np.float32).reshape(payload.shape), config
    )


def validate_array(array: np.ndarray, config: APIConfig) -> np.ndarray:
    if array.ndim != 3:
        raise APIError("API-VALID-422", "Array must be 3D.", status_code=422)
    if array.nbytes > config.maximum_array_bytes:
        raise APIError("API-SIZE-413", "Array exceeds configured size limit.", status_code=413)
    if not np.all(np.isfinite(array)):
        raise APIError("API-VALID-422", "Array contains non-finite values.", status_code=422)
    return array.astype(np.float32)


def safe_output_child(root: Path, child: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / child).resolve()
    if os.path.commonpath([str(resolved_root), str(path)]) != str(resolved_root):
        raise APIError("API-AUTH-403", "Output path escapes configured root.", status_code=403)
    return path


def public_path(path: Path) -> str:
    """Return a non-absolute display path."""
    return path.name if path.is_file() else str(Path(path.name))
