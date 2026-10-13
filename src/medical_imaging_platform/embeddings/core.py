"""Embedding normalization, persistence, and exact search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from medical_imaging_platform.embeddings.config import EmbeddingConfig


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
        # Persisted and in-memory indexes share the same stored-vector contract.
        validate_matrix(matrix)
        self.matrix = matrix
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
    def load(cls, directory: Path, *, maximum_bytes: int | None = None) -> EmbeddingIndex:
        directory = directory.resolve()
        manifest_path = directory / "embedding_manifest.json"
        matrix_path = directory / "embedding_matrix.npy"
        for path in (manifest_path, matrix_path, directory / "evidence" / "checksum_manifest.json"):
            if path.is_symlink() or not path.resolve().is_relative_to(directory):
                raise ValueError("Embedding artifact escapes index directory")
            if maximum_bytes is not None and path.exists() and path.stat().st_size > maximum_bytes:
                raise ValueError("Embedding artifact exceeds configured size limit")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Embedding manifest must be an object")
        expected = manifest.get("matrix_checksum")
        if not expected or sha256_file(matrix_path) != expected:
            raise ValueError("Embedding matrix checksum mismatch")
        if not isinstance(manifest, dict):
            raise ValueError("Embedding manifest must be an object")
        matrix = np.load(matrix_path, allow_pickle=False)
        validate_matrix(matrix)
        if (
            type(manifest.get("embedding_dimension")) is not int
            or manifest.get("embedding_dimension") != matrix.shape[1]
        ):
            raise ValueError("Embedding dimension does not match manifest")
        for key in (
            "encoder_checkpoint_checksum",
            "config_checksum",
            "source_dataset_manifest_checksum",
        ):
            digest = manifest.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"Invalid compatibility metadata: {key}")
        if (
            manifest.get("encoder_family") not in {"small_3d_cnn", "tiny_3d_vit"}
            or manifest.get("normalization") != "l2"
        ):
            raise ValueError("Invalid encoder/normalization compatibility metadata")
        records = manifest.get("records", [])
        if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
            raise ValueError("Manifest records must be objects")
        keys = [(r.get("source_sample_id", ""), r.get("embedding_id", "")) for r in records]
        if any(not isinstance(sid, str) or not isinstance(eid, str) for sid, eid in keys):
            raise ValueError("Manifest row identifiers must be strings")
        if (
            any(not sid or not eid for sid, eid in keys)
            or keys != sorted(keys)
            or len({eid for _, eid in keys}) != len(keys)
        ):
            raise ValueError("Invalid manifest row ordering or duplicate embedding IDs")
        for record in records:
            for key in ("encoder_family", "encoder_checkpoint_checksum"):
                if key in record and record[key] != manifest[key]:
                    raise ValueError("Record encoder/checkpoint compatibility mismatch")
        evidence = directory / "evidence" / "checksum_manifest.json"
        if not evidence.exists():
            raise ValueError("Missing evidence/checksum_manifest.json")
        files = json.loads(evidence.read_text(encoding="utf-8")).get("files", {})
        if not {"embedding_matrix.npy", "embedding_manifest.json"} <= files.keys():
            raise ValueError("Incomplete index evidence checksums")
        for name, digest in files.items():
            path = directory / name
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(directory)
                or sha256_file(path) != digest
            ):
                raise ValueError("Index evidence checksum mismatch")
        index = cls(matrix, manifest)
        policy = index_policy(index)
        if "retrieval_policy" in manifest and (
            manifest.get("gallery_splits", policy.gallery_splits) != policy.gallery_splits
            or manifest.get("query_splits", policy.query_splits) != policy.query_splits
        ):
            raise ValueError("Manifest split metadata contradicts retrieval policy")
        return index

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
                    "encoder_family": record.get(
                        "encoder_family", self.manifest.get("encoder_family")
                    ),
                    "checkpoint_checksum": record.get(
                        "encoder_checkpoint_checksum",
                        self.manifest.get("encoder_checkpoint_checksum"),
                    ),
                    "source_checksum": record.get(
                        "source_checksum", record.get("source_volume_checksum")
                    ),
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


def validate_matrix(matrix: np.ndarray) -> None:
    """Verify the stored bytes' numerical contract without changing the array."""
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings must be finite")
    if matrix.dtype != np.float32:
        raise ValueError("Persisted embeddings must have dtype float32")
    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError("Persisted embeddings must have positive rank-2 dimensions")
    if not np.allclose(np.linalg.norm(matrix.astype(np.float64), axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Stored embedding L2 normalization tolerance failure")


def index_policy(index: EmbeddingIndex) -> EmbeddingConfig:
    """Use the persisted policy, with explicit legacy defaults."""
    return EmbeddingConfig.model_validate(
        index.manifest.get(
            "retrieval_policy",
            {
                "policy_version": "M22-v1",
                "gallery_splits": index.manifest.get("gallery_splits", ["train", "validation"]),
                "query_splits": index.manifest.get("query_splits", ["test"]),
                "top_k": [1, 5],
                "relevance": "scenario",
                "exclude_self": True,
                "exclude_same_source": False,
                "output_directory": "ml/embeddings",
            },
        )
    )


def query_index(
    index: EmbeddingIndex, sample_id: str, top_k: int, policy: EmbeddingConfig | None = None
) -> list[dict[str, Any]]:
    policy = policy or index_policy(index)
    if sample_id not in index.sample_index:
        raise ValueError("query sample_id is not present in the embedding index")
    row = index.sample_index[sample_id]
    record = index.records[row]
    return index.search(
        index.matrix[row],
        top_k,
        exclude_embedding_id=record["embedding_id"] if policy.exclude_self else None,
        exclude_source_sample_id=record["source_sample_id"] if policy.exclude_same_source else None,
        gallery_splits=tuple(policy.gallery_splits),
    )
