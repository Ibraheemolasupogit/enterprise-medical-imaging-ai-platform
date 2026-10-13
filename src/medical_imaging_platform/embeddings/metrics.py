"""Deterministic retrieval metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _average_precision(relevant: list[bool], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    hits = 0
    score = 0.0
    for rank, is_relevant in enumerate(relevant[:k], start=1):
        if is_relevant:
            hits += 1
            score += hits / rank
    return score / min(total_relevant, k)


def retrieval_metrics(
    results: Iterable[dict[str, Any]], relevance: dict[str, set[str]], ks: tuple[int, ...] = (1, 5)
) -> dict[str, Any]:
    rows = list(results)
    relevant = relevance.get("sample_ids", set())
    scenario_relevant = relevance.get("scenarios", set())
    label_relevant = relevance.get("labels", set())
    out: dict[str, Any] = {}
    for k in ks:
        top = rows[:k]
        hits = [str(row.get("source_sample_id")) in relevant for row in top]
        out[f"recall_at_{k}"] = float(sum(hits) / len(relevant)) if relevant else 0.0
        out[f"precision_at_{k}"] = float(sum(hits) / k) if k else 0.0
        out[f"map_at_{k}"] = _average_precision(hits, len(relevant), k)
        out[f"scenario_consistency_at_{k}"] = (
            float(sum(str(row.get("scenario")) in scenario_relevant for row in top) / len(top))
            if top
            else 0.0
        )
        out[f"label_consistency_at_{k}"] = (
            float(sum(str(row.get("label")) in label_relevant for row in top) / len(top))
            if top
            else 0.0
        )
    out["mrr"] = next(
        (
            1.0 / rank
            for rank, row in enumerate(rows, 1)
            if str(row.get("source_sample_id")) in relevant
        ),
        0.0,
    )
    out["gallery_size"] = int(len(rows))
    return out


def evaluate_index(index: Any, policy: Any) -> dict[str, Any]:
    """One relevance and candidate policy shared by evaluation and benchmarking."""
    from medical_imaging_platform.embeddings.core import query_index

    queries = [r for r in index.records if r.get("split") in policy.query_splits]
    metrics = []
    for query in queries:
        hits = query_index(index, query["source_sample_id"], max(policy.top_k), policy)
        candidates = [
            r
            for r in index.records
            if r.get("split") in policy.gallery_splits
            and (not policy.exclude_self or r["embedding_id"] != query["embedding_id"])
            and (
                not policy.exclude_same_source or r["source_sample_id"] != query["source_sample_id"]
            )
        ]
        relevant = {
            r["source_sample_id"]
            for r in candidates
            if r.get(policy.relevance) == query.get(policy.relevance)
        }
        metrics.append(
            retrieval_metrics(
                hits,
                {
                    "sample_ids": relevant,
                    "scenarios": {str(query.get("scenario"))},
                    "labels": {str(query.get("label"))},
                },
                policy.top_k,
            )
        )
    return {
        "query_count": len(queries),
        "top_k": list(policy.top_k),
        "policy": policy.model_dump(mode="json"),
        **(
            {key: sum(float(row[key]) for row in metrics) / len(metrics) for key in metrics[0]}
            if metrics
            else {}
        ),
    }
