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


def evaluate_query(index: Any, query_record: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    index.records[index.record_index[query_record["embedding_id"]]]
    result = index.search(
        index.matrix[index.record_index[query_record["embedding_id"]]],
        top_k=top_k,
        exclude_embedding_id=query_record["embedding_id"],
    )
    relevant = {
        "sample_ids": set(query_record.get("relevant_sample_ids", [])),
        "scenarios": {str(query_record.get("scenario"))},
        "labels": {str(query_record.get("label"))},
    }
    return retrieval_metrics(result, relevant, ks=(1, top_k))
