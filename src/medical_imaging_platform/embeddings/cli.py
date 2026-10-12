# mypy: disable-error-code="attr-defined"
"""M22 embedding and retrieval command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_embedding_benchmark
from .config import load_embedding_config, validate_top_k
from .core import EmbeddingIndex, validate_index
from .extractor import generate_embeddings
from .metrics import retrieval_metrics


def _positive_k(value: str) -> int:
    k = int(value)
    if k <= 0:
        raise argparse.ArgumentTypeError("top_k must be a positive integer")
    return k


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medical-imaging-platform embeddings")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate-embeddings")
    p.add_argument("dataset_dir", type=Path)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--encoder-family", choices=("small_3d_cnn", "tiny_3d_vit"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("query-similar-studies")
    p.add_argument("index_dir", type=Path)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--top-k", type=_positive_k, default=5)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("validate-embedding-index")
    p.add_argument("index_dir", type=Path)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("evaluate-retrieval")
    p.add_argument("index_dir", type=Path)
    p.add_argument("--config", type=Path, default=Path("config/embeddings.yaml"))
    p.add_argument("--top-k", type=_positive_k, nargs="+", default=None)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("run-embedding-benchmark")
    p.add_argument("--cnn-index", type=Path, required=True)
    p.add_argument("--vit-index", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("config/embeddings.yaml"))
    p.add_argument("--top-k", type=_positive_k, nargs="+", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--json", action="store_true")
    return parser


def _evaluate(index: EmbeddingIndex, top_k: tuple[int, ...]) -> dict[str, object]:
    validate_top_k(top_k)
    queries = [r for r in index.records if r.get("split") == "test"]
    per = []
    for query in queries:
        hits = index.search(
            index.matrix[index.embedding_index[query["embedding_id"]]],
            top_k=max(top_k),
            exclude_embedding_id=query["embedding_id"],
            gallery_splits=tuple(index.manifest.get("gallery_splits", ["train", "validation"])),
        )
        relevant = {
            r["source_sample_id"]
            for r in index.records
            if r.get("scenario") == query.get("scenario")
            and r.get("split") in {"train", "validation"}
        }
        per.append(
            retrieval_metrics(
                hits,
                {
                    "sample_ids": relevant,
                    "scenarios": {str(query.get("scenario"))},
                    "labels": {str(query.get("label"))},
                },
                ks=top_k,
            )
        )
    if not per:
        return {"query_count": 0, "top_k": list(top_k)}
    return {
        "query_count": len(per),
        "top_k": list(top_k),
        **{key: sum(float(item.get(key, 0.0)) for item in per) / len(per) for key in per[0]},
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"evaluate-retrieval", "run-embedding-benchmark"}:
        policy = load_embedding_config(args.config)
        try:
            # CLI collections become the canonical tuple at this boundary only.
            args.top_k = validate_top_k(tuple(args.top_k) if args.top_k else policy.top_k)
        except ValueError as exc:
            _parser().error(str(exc))
    if args.command == "generate-embeddings":
        result = generate_embeddings(
            args.dataset_dir,
            args.config,
            args.checkpoint,
            args.encoder_family,
            args.output_dir,
            args.overwrite,
        )
    elif args.command == "validate-embedding-index":
        result = validate_index(args.index_dir)
    elif args.command == "query-similar-studies":
        index = EmbeddingIndex.load(args.index_dir)
        if args.sample_id not in index.sample_index:
            raise SystemExit(f"Unknown sample ID: {args.sample_id}")
        i = index.sample_index[args.sample_id]
        record = index.records[i]
        result = {
            "query": {
                "sample_id": args.sample_id,
                "encoder_family": index.manifest.get("encoder_family"),
                "index_checksum": index.manifest.get("matrix_checksum"),
                "research_only": True,
            },
            "results": index.search(
                index.matrix[i], args.top_k, exclude_embedding_id=record["embedding_id"]
            ),
        }
    elif args.command == "evaluate-retrieval":
        result = _evaluate(EmbeddingIndex.load(args.index_dir), args.top_k)
    else:
        result = run_embedding_benchmark(
            cnn_index=args.cnn_index,
            vit_index=args.vit_index,
            output_dir=args.output_dir,
            top_k=args.top_k,
            overwrite=args.overwrite,
        )
    print(
        json.dumps(result, indent=2, sort_keys=True, default=str)
        if getattr(args, "json", False)
        else result
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
