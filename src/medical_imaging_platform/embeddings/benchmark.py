# mypy: disable-error-code="has-type"
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from medical_imaging_platform.embeddings.config import EmbeddingConfig, validate_top_k
from medical_imaging_platform.embeddings.core import EmbeddingIndex, index_policy, sha256_file
from medical_imaging_platform.embeddings.metrics import evaluate_index


def run_benchmark(
    cnn_index: str | Path,
    vit_index: str | Path,
    output_dir: str | Path,
    *,
    top_k: tuple[int, ...] | None = None,
    policy: EmbeddingConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if top_k is not None:
        validate_top_k(top_k)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cnn = EmbeddingIndex.load(Path(cnn_index))
    vit = EmbeddingIndex.load(Path(vit_index))
    cnn_ids = [r["source_sample_id"] for r in cnn.records]
    vit_ids = [r["source_sample_id"] for r in vit.records]
    if cnn_ids != vit_ids:
        raise ValueError("CNN and ViT indexes do not have identical ordered samples")
    membership_keys = ("split", "research_subject_id", "scenario", "label")
    if [[r.get(key) for key in membership_keys] for r in cnn.records] != [
        [r.get(key) for key in membership_keys] for r in vit.records
    ] or cnn.manifest.get("source_dataset_manifest_checksum") != vit.manifest.get(
        "source_dataset_manifest_checksum"
    ):
        raise ValueError("CNN and ViT indexes do not have identical source dataset and membership")
    if policy is None and index_policy(cnn) != index_policy(vit):
        raise ValueError("CNN and ViT relevance/retrieval policies are incompatible")
    policy = policy or index_policy(cnn)
    if top_k is not None:
        policy = policy.model_copy(update={"top_k": top_k})
    top_k = policy.top_k
    query_ids = [
        r["source_sample_id"] for r in cnn.records if r.get("split") in policy.query_splits
    ]
    gallery_ids = {
        r["source_sample_id"] for r in cnn.records if r.get("split") in policy.gallery_splits
    }
    results: dict[str, Any] = {}
    for name, index in (("small_3d_cnn", cnn), ("tiny_3d_vit", vit)):
        started = time.perf_counter()
        evaluation = evaluate_index(index, policy)
        elapsed = time.perf_counter() - started
        numeric = {
            key: value
            for key, value in evaluation.items()
            if isinstance(value, (int, float)) and key != "query_count"
        }
        results[name] = {
            "metrics": numeric,
            "embedding_dimension": index.dimension,
            "matrix_storage_bytes": int(index.matrix.nbytes),
            "query_count": len(query_ids),
            "query_latency_seconds": elapsed / max(len(query_ids), 1),
            "checkpoint_checksum": index.manifest.get("encoder_checkpoint_checksum"),
        }
    comparison = {
        "models": results,
        "retrieval_policy": policy.model_dump(mode="json"),
        "fairness_attestation": {
            "identical_ordered_samples": True,
            "gallery_sample_count": len(gallery_ids),
            "query_sample_count": len(query_ids),
            "top_k": list(top_k),
            "similarity": "exact cosine via dot product on L2-normalized vectors",
        },
    }
    manifest = {
        "retrieval_policy": policy.model_dump(mode="json"),
        "cnn_index_checksum": sha256_file(Path(cnn_index) / "embedding_manifest.json"),
        "vit_index_checksum": sha256_file(Path(vit_index) / "embedding_manifest.json"),
        "gallery_sample_ids": sorted(gallery_ids),
        "query_sample_ids": query_ids,
        "top_k": list(top_k),
    }
    nl = chr(10)
    (output_dir / "embedding_benchmark_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + nl
    )
    (output_dir / "embedding_benchmark_comparison.json").write_text(
        json.dumps(comparison, sort_keys=True, indent=2) + nl
    )
    report = (
        "# M22 Embedding Benchmark"
        + "\n\n"
        + "M22 compares encoder representations on fixed samples; this is not a clinical "
        + "similarity assessment.\n"
    )
    (output_dir / "embedding_benchmark_report.md").write_text(report)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    (evidence_dir / "checksum_manifest.json").write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "files": {
                    name: sha256_file(output_dir / name)
                    for name in (
                        "embedding_benchmark_manifest.json",
                        "embedding_benchmark_comparison.json",
                        "embedding_benchmark_report.md",
                    )
                },
            },
            sort_keys=True,
            indent=2,
        )
        + nl
    )
    return comparison


__all__ = ["run_benchmark"]


run_embedding_benchmark = run_benchmark
