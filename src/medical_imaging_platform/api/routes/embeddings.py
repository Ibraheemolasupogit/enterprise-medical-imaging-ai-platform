# mypy: disable-error-code="misc"
"""Governed exact-search retrieval route for M22 embeddings."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from medical_imaging_platform.api.dependencies import get_api_config
from medical_imaging_platform.api.errors import APIError
from medical_imaging_platform.api.models import APIConfig
from medical_imaging_platform.embeddings.config import load_embedding_config
from medical_imaging_platform.embeddings.core import (
    EmbeddingIndex,
    index_policy,
    query_index,
    sha256_file,
)
from medical_imaging_platform.evidence import ArtifactBinding

router = APIRouter(prefix="/v1/embeddings", tags=["embeddings"])


class SearchRequest(BaseModel):
    """Bounded request for a governed embedding-index search."""

    model_config = ConfigDict(extra="forbid")
    index_directory: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    top_k: StrictInt = Field(default=5, ge=1, le=100)


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
        index = EmbeddingIndex.load(index_directory, maximum_bytes=config.maximum_array_bytes)
        policy = (
            load_embedding_config(config.embedding_policy)
            if config.embedding_policy
            else index_policy(index)
        )
        if request.sample_id not in index.sample_index:
            raise ValueError("query sample_id is not present in the embedding index")
        query_record = index.records[index.sample_index[request.sample_id]]
        results = query_index(index, request.sample_id, request.top_k, policy)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        raise APIError(
            "API-VALID-400",
            str(exc)
            if isinstance(exc, ValueError)
            else "Embedding index is unavailable or invalid.",
            status_code=400,
        ) from exc
    return {
        "query": query_record,
        "retrieval_policy": policy.model_dump(mode="json"),
        "artifact": ArtifactBinding(
            artifact_type="embedding_index",
            artifact_reference=str(index_directory / "embedding_manifest.json"),
            artifact_checksum=sha256_file(index_directory / "embedding_manifest.json"),
            run_id=str(index.manifest.get("index_run_id", index.manifest["matrix_checksum"][:16])),
            source_checksum=index.manifest["source_dataset_manifest_checksum"],
            model_family=index.manifest["encoder_family"],
            checkpoint_checksum=index.manifest["encoder_checkpoint_checksum"],
        ).model_dump(mode="json"),
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


@router.get("/indexes")
def list_indexes(config: Annotated[APIConfig, Depends(get_api_config)]) -> dict[str, object]:
    """List explicitly configured indexes; never scan arbitrary filesystem roots."""
    entries = []
    for directory in config.embedding_indexes:
        entry: dict[str, object] = {"index_directory": str(directory), "status": "unavailable"}
        try:
            if not _under_allowed_root(directory, config.allowed_input_roots):
                raise ValueError("Index is outside allowed input roots")
            index = EmbeddingIndex.load(directory, maximum_bytes=config.maximum_array_bytes)
            policy = (
                load_embedding_config(config.embedding_policy)
                if config.embedding_policy
                else index_policy(index)
            )
            entry.update(
                status="available",
                encoder_family=index.manifest["encoder_family"],
                sample_ids=[
                    r["source_sample_id"]
                    for r in index.records
                    if r.get("split") in policy.query_splits
                ],
            )
        except (OSError, ValueError, KeyError):
            entry["warning"] = "Configured embedding index is unavailable or invalid."
        entries.append(entry)
    return {"indexes": entries}
