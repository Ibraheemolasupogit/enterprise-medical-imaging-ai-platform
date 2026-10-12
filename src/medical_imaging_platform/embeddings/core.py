"""Embedding normalization, persistence, and exact search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Embeddings must be finite rank-2 numeric values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero-norm embeddings are not valid for retrieval")
    return (array / norms).astype(np.float32)  # type: ignore[no-any-return]


class EmbeddingIndex:
    def __init__(self, matrix: np.ndarray, manifest: dict[str, Any]) -> None:
        self.matrix = l2_normalize(matrix)
        self.manifest = manifest
        self.records: list[dict[str, Any]] = list(manifest.get("records", []))
        if len(self.records) != self.matrix.shape[0]:
            raise ValueError("Embedding matrix rows do not match manifest records")
        self.sample_index: dict[str, int] = {
            str(
                record.get("sample_id", record.get("source_sample_id", record.get("embedding_id")))
            ): i
            for i, record in enumerate(self.records)
        }
        self.embedding_index: dict[str, int] = {
            str(record["embedding_id"]): i for i, record in enumerate(self.records)
        }

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[1])

    @classmethod
    def load(cls, directory: Path) -> EmbeddingIndex:
        manifest_path = directory / "embedding_manifest.json"
        matrix_path = directory / "embedding_matrix.npy"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        matrix = np.load(matrix_path, allow_pickle=False)
        expected = manifest.get("matrix_checksum")
        if expected and sha256_file(matrix_path) != expected:
            raise ValueError("Embedding matrix checksum mismatch")
        return cls(matrix, manifest)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
        exclude_embedding_id: str | None = None,
        exclude_source_sample_id: str | None = None,
        gallery_splits: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        q = l2_normalize(query)[0]
        if q.shape[0] != self.matrix.shape[1]:
            raise ValueError("Query dimension does not match embedding index")
        scores = self.matrix @ q
        candidates = []
        if gallery_splits is None and "gallery_splits" in self.manifest:
            gallery_splits = tuple(self.manifest["gallery_splits"])
        for i, record in enumerate(self.records):
            if gallery_splits is not None and record.get("split") not in gallery_splits:
                continue
            if exclude_embedding_id and record.get("embedding_id") == exclude_embedding_id:
                continue
            if (
                exclude_source_sample_id
                and record.get("source_sample_id") == exclude_source_sample_id
            ):
                continue
            candidates.append(i)
        candidates.sort(key=lambda i: (-float(scores[i]), str(self.records[i].get("embedding_id"))))
        results = []
        for rank, i in enumerate(candidates[:top_k], start=1):
            record = self.records[i]
            results.append(
                {
                    "rank": rank,
                    "embedding_id": record.get("embedding_id"),
                    "source_sample_id": record.get("source_sample_id"),
                    "similarity": float(scores[i]),
                    "scenario": record.get("scenario"),
                    "label": record.get("label"),
                    "encoder_family": record.get("encoder_family"),
                    "checkpoint_checksum": record.get("encoder_checkpoint_checksum"),
                    "source_checksum": record.get("source_volume_checksum"),
                }
            )
        return results


def validate_index(directory: Path) -> dict[str, Any]:
    index = EmbeddingIndex.load(directory)
    evidence = directory / "evidence" / "checksum_manifest.json"
    if not evidence.exists():
        raise ValueError("Missing evidence/checksum_manifest.json")
    return {
        "valid": True,
        "rows": int(index.matrix.shape[0]),
        "dimension": int(index.matrix.shape[1]),
        "dtype": str(index.matrix.dtype),
        "normalized": True,
    }
