"""Reviewer UI security and upload validation."""

from __future__ import annotations

import io
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from medical_imaging_platform.reviewer_ui.models import ReviewerUIConfig


class ReviewerUISecurityError(ValueError):
    """Raised when UI input fails local security controls."""


def enforce_loopback_url(url: str, *, allow_remote: bool) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ReviewerUISecurityError("API URL must use http or https.")
    if not allow_remote and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ReviewerUISecurityError("Remote API URLs are disabled by policy.")
    return url.rstrip("/")


def validate_evidence_id(value: str) -> str:
    if "/" in value or "\\" in value or ".." in value or not value.strip():
        raise ReviewerUISecurityError("Evidence identifiers must not contain path traversal.")
    return value.strip()


def validate_upload_metadata(filename: str, size_bytes: int, config: ReviewerUIConfig) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in config.allowed_upload_extensions:
        raise ReviewerUISecurityError("Unsupported upload extension.")
    if size_bytes > config.maximum_upload_bytes:
        raise ReviewerUISecurityError("Upload exceeds configured size limit.")


def load_uploaded_npy(
    filename: str,
    content: bytes,
    config: ReviewerUIConfig,
    *,
    expected_ndim: int = 3,
) -> np.ndarray:
    validate_upload_metadata(filename, len(content), config)
    if Path(filename).suffix.lower() != ".npy":
        raise ReviewerUISecurityError("Array uploads must be .npy files.")
    try:
        array = np.load(io.BytesIO(content), allow_pickle=False)
    except ValueError as exc:
        raise ReviewerUISecurityError("Invalid NumPy upload.") from exc
    if array.ndim != expected_ndim:
        raise ReviewerUISecurityError("Uploaded array has an invalid shape.")
    if not np.all(np.isfinite(array)):
        raise ReviewerUISecurityError("Uploaded array contains non-finite values.")
    return np.asarray(array, dtype=np.float32)


def array_payload(array: np.ndarray) -> dict[str, object]:
    return {"shape": list(array.shape), "values": array.astype(float).ravel().tolist()}


def safe_review_output_dir(root: Path, review_id: str) -> Path:
    if "/" in review_id or "\\" in review_id or ".." in review_id or not review_id:
        raise ReviewerUISecurityError("Review ID is not safe for export.")
    resolved_root = root.resolve()
    target = (resolved_root / review_id).resolve()
    if os.path.commonpath([str(resolved_root), str(target)]) != str(resolved_root):
        raise ReviewerUISecurityError("Review export path escapes configured output root.")
    return target
