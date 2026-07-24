"""Command-line interface for repository foundation and synthetic data tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from medical_imaging_platform import __version__
from medical_imaging_platform.synthetic.generator import load_synthetic_config
from medical_imaging_platform.synthetic.io import generate_dataset, validate_dataset
from medical_imaging_platform.utils.config import ConfigError, validate_repository_configs
from medical_imaging_platform.utils.exceptions import MedicalImagingPlatformError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medical-imaging-platform",
        description="Repository foundation utilities for the medical imaging platform.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="Print the package version.")

    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Validate repository YAML configuration files.",
    )
    validate_parser.add_argument(
        "--config-dir",
        default="config",
        help="Configuration directory to validate. Defaults to ./config.",
    )

    generate_parser = subparsers.add_parser(
        "generate-synthetic-data",
        help="Generate deterministic synthetic CT-like engineering fixtures.",
    )
    generate_parser.add_argument(
        "--config",
        default="config/data.yaml",
        help="Path to data configuration. Defaults to config/data.yaml.",
    )
    generate_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to settings.synthetic_data.output_root.",
    )
    generate_parser.add_argument(
        "--cases",
        type=int,
        default=None,
        help="Number of cases to generate. Defaults to configured dataset_size.",
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Defaults to configured random_seed.",
    )
    generate_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )

    dataset_parser = subparsers.add_parser(
        "validate-dataset",
        help="Validate a generated synthetic dataset manifest and checksums.",
    )
    dataset_parser.add_argument("dataset_dir", help="Synthetic dataset directory to validate.")

    summary_parser = subparsers.add_parser(
        "summarise-dataset",
        help="Print summary statistics for a generated synthetic dataset.",
    )
    summary_parser.add_argument("dataset_dir", help="Synthetic dataset directory to summarise.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "validate-config":
        try:
            validated = validate_repository_configs(Path(args.config_dir))
        except ConfigError as exc:
            print(f"Configuration validation failed: {exc}")
            return 1
        print(f"Validated {len(validated.configs)} configuration files.")
        return 0

    if args.command == "generate-synthetic-data":
        try:
            config = load_synthetic_config(Path(args.config))
            output_dir = (
                Path(args.output_dir) if args.output_dir is not None else config.output_root
            )
            manifest = generate_dataset(
                config=config,
                output_root=output_dir,
                case_count=args.cases,
                random_seed=args.seed,
                overwrite=args.overwrite,
            )
        except MedicalImagingPlatformError as exc:
            print(f"Synthetic dataset generation failed: {exc}")
            return 1
        print(
            f"Generated {len(manifest.records)} synthetic cases at {output_dir} "
            f"with dataset_id={manifest.dataset_id}."
        )
        return 0

    if args.command == "validate-dataset":
        try:
            manifest = validate_dataset(Path(args.dataset_dir))
        except MedicalImagingPlatformError as exc:
            print(f"Dataset validation failed: {exc}")
            return 1
        print(f"Validated synthetic dataset with {len(manifest.records)} cases.")
        return 0

    if args.command == "summarise-dataset":
        try:
            manifest = validate_dataset(Path(args.dataset_dir))
        except MedicalImagingPlatformError as exc:
            print(f"Dataset summary failed: {exc}")
            return 1
        print(manifest.summary)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
