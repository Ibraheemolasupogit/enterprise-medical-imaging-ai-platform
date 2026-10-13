# mypy: disable-error-code="attr-defined"
"""M22 embedding and retrieval command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_embedding_benchmark
from .config import EmbeddingConfig, load_embedding_config, validate_top_k
from .core import EmbeddingIndex, index_policy, query_index, validate_index
from .extractor import generate_embeddings
from .metrics import evaluate_index


def _positive_k(value: str) -> int:
    k = int(value)
    if k <= 0:
        raise argparse.ArgumentTypeError("top_k must be a positive integer")
    return k


COMMANDS = {
    "generate-embeddings",
    "query-similar-studies",
    "validate-embedding-index",
    "evaluate-retrieval",
    "run-embedding-benchmark",
}


def register_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register embedding arguments once on the platform parser."""
    p = sub.add_parser("generate-embeddings")
    p.add_argument("dataset_dir", type=Path)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--policy-config", type=Path, default=Path("config/embeddings.yaml"))
    p.add_argument("--encoder-family", choices=("small_3d_cnn", "tiny_3d_vit"), required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("query-similar-studies")
    p.add_argument("index_dir", type=Path)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--config", type=Path)
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


def _parser() -> argparse.ArgumentParser:
    """Compatibility access to the shared platform parser."""
    from medical_imaging_platform.cli import build_parser

    return build_parser()


def _evaluate(
    index: EmbeddingIndex, top_k: tuple[int, ...], policy: EmbeddingConfig | None = None
) -> dict[str, object]:
    validate_top_k(top_k)
    policy = (policy or index_policy(index)).model_copy(update={"top_k": top_k})
    return evaluate_index(index, policy)


def handle(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Execute already parsed arguments."""
    if args.command in {"evaluate-retrieval", "run-embedding-benchmark"}:
        policy = load_embedding_config(args.config)
        try:
            # CLI collections become the canonical tuple at this boundary only.
            args.top_k = validate_top_k(tuple(args.top_k) if args.top_k else policy.top_k)
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "generate-embeddings":
        policy = load_embedding_config(args.policy_config)
        result = generate_embeddings(
            args.dataset_dir,
            args.config,
            args.checkpoint,
            args.encoder_family,
            args.output_dir or Path(policy.output_directory),
            args.overwrite,
            policy=policy,
        )
    elif args.command == "validate-embedding-index":
        result = validate_index(args.index_dir)
    elif args.command == "query-similar-studies":
        index = EmbeddingIndex.load(args.index_dir)
        policy = load_embedding_config(args.config) if args.config else index_policy(index)
        if args.sample_id not in index.sample_index:
            raise SystemExit(f"Unknown sample ID: {args.sample_id}")
        result = {
            "query": {
                "sample_id": args.sample_id,
                "encoder_family": index.manifest.get("encoder_family"),
                "index_checksum": index.manifest.get("matrix_checksum"),
                "research_only": True,
            },
            "results": query_index(index, args.sample_id, args.top_k, policy),
            "retrieval_policy": policy.model_dump(mode="json"),
        }
    elif args.command == "evaluate-retrieval":
        result = _evaluate(EmbeddingIndex.load(args.index_dir), args.top_k, policy)
    else:
        result = run_embedding_benchmark(
            cnn_index=args.cnn_index,
            vit_index=args.vit_index,
            output_dir=args.output_dir,
            top_k=args.top_k,
            policy=policy,
            overwrite=args.overwrite,
        )
    print(
        json.dumps(result, indent=2, sort_keys=True, default=str)
        if getattr(args, "json", False)
        else result
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Keep the historical embedding entry point on the same command surface."""
    from medical_imaging_platform.cli import main as platform_main

    return platform_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
