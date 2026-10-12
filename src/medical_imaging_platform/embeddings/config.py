"""Strict M22 embedding policy configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


def validate_top_k(value: tuple[int, ...]) -> tuple[int, ...]:
    """Require the canonical ordered collection used for evaluation and benchmarks."""
    if (
        not isinstance(value, tuple)
        or not value
        or any(type(k) is not int or k <= 0 for k in value)
        or value != tuple(sorted(set(value)))
    ):
        raise ValueError("top_k must be a non-empty tuple of unique, sorted, positive integers")
    return value


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_version: str = Field(min_length=1)
    normalization: Literal["l2"] = "l2"
    gallery_splits: list[str] = Field(min_length=1)
    query_splits: list[str] = Field(min_length=1)
    top_k: tuple[StrictInt, ...] = Field(min_length=1)
    relevance: Literal["scenario", "label"] = "scenario"
    exclude_self: bool = True
    exclude_same_source: bool = False
    output_directory: str = Field(min_length=1)
    compatibility_policy: Literal["strict"] = "strict"

    @field_validator("top_k")
    @classmethod
    def valid_k(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        return validate_top_k(value)

    @field_validator("gallery_splits", "query_splits")
    @classmethod
    def valid_splits(cls, value: list[str]) -> list[str]:
        if any(not split for split in value):
            raise ValueError("split names must be non-empty")
        return value


def load_embedding_config(path: str | Path) -> EmbeddingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return EmbeddingConfig.model_validate(raw)
