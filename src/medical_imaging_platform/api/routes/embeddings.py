# mypy: disable-error-code="misc"
"""Governed exact-search retrieval route for M22 embeddings."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from medical_imaging_platform.api.dependencies import get_api_config
from medical_imaging_platform.api.models import APIConfig
from medical_imaging_platform.embeddings.core import EmbeddingIndex

router = APIRouter(prefix="/v1/embeddings", tags=["embeddings"])


class SearchRequest(BaseModel):
    """Bounded request for a governed embedding-index search."""

    model_config = ConfigDict(extra="forbid")
    index_directory: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


def _under_allowed_root(path: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root.resolve() or root.resolve() in resolved.parents for root in roots)


@router.post("/search")
def search_embeddings(
    request: SearchRequest, config: Annotated[APIConfig, Depends(get_api_config)]
) -> dict[str, object]:
    """Search a validated, locally governed embedding index."""
    index_directory = Path(request.index_directory)
    if not _under_allowed_root(index_directory, config.allowed_input_roots):
        raise HTTPException(
            status_code=403, detail="embedding index is outside allowed input roots"
        )
    started = time.perf_counter()
    try:
        index = EmbeddingIndex.load(index_directory)
        records = index.manifest.get("records", [])
        query_record = next(
            (
                record
                for record in records
                if record.get("sample_id") == request.sample_id
                or record.get("source_sample_id") == request.sample_id
            ),
            None,
        )
        if query_record is None:
            raise ValueError("query sample_id is not present in the embedding index")
        embedding_id = str(query_record["embedding_id"])
        row_index = next(
            index_value
            for index_value, record in enumerate(records)
            if str(record["embedding_id"]) == embedding_id
        )
        results = index.search(
            index.matrix[row_index], top_k=request.top_k, exclude_embedding_id=embedding_id
        )
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "query": query_record,
        "results": results,
        "encoder_family": index.manifest.get("encoder_family"),
        "encoder_checkpoint_checksum": index.manifest.get("encoder_checkpoint_checksum"),
        "index_checksum": index.manifest.get("matrix_checksum"),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "warnings": [
            "Embedding-space similarity is not diagnostic similarity.",
            "Synthetic/de-identified research and engineering use only; "
            "human interpretation is required.",
        ],
    }
