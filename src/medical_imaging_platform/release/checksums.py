"""Checksum helpers for release evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return a SHA-256 hex digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_paths(paths: list[Path]) -> dict[str, str]:
    """Return stable relative-path checksums for existing files."""
    return {path.as_posix(): sha256_file(path) for path in sorted(paths)}
