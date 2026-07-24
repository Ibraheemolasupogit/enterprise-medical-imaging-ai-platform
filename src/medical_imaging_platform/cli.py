"""Command-line interface for repository foundation tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from medical_imaging_platform import __version__
from medical_imaging_platform.utils.config import ConfigError, validate_repository_configs


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

    parser.error(f"Unsupported command: {args.command}")
    return 2
