"""Command-line interface for repository foundation and synthetic data tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical_imaging_platform import __version__
from medical_imaging_platform.deidentification.deidentifier import deidentify_series
from medical_imaging_platform.deidentification.policy import default_policy
from medical_imaging_platform.ingestion.discovery import discover_dicom_series
from medical_imaging_platform.ingestion.fixtures import generate_dicom_fixture_series
from medical_imaging_platform.ingestion.loader import load_dicom
from medical_imaging_platform.ingestion.metadata import extract_metadata
from medical_imaging_platform.ingestion.validation import validate_series
from medical_imaging_platform.preprocessing.errors import (
    PreprocessingError,
    PreprocessingOutputError,
    PreprocessingQualityError,
    PreprocessingRejectedError,
)
from medical_imaging_platform.preprocessing.export import (
    inspect_preprocessed_volume,
    validate_preprocessed_volume,
)
from medical_imaging_platform.preprocessing.pipeline import preprocess_dicom_series
from medical_imaging_platform.quality_control.pipeline import run_quality_control
from medical_imaging_platform.synthetic.generator import load_synthetic_config
from medical_imaging_platform.synthetic.io import generate_dataset, validate_dataset
from medical_imaging_platform.utils.config import (
    ConfigError,
    load_dicom_ingestion_config,
    load_preprocessing_config,
    load_quality_control_config,
    validate_repository_configs,
)
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

    fixture_parser = subparsers.add_parser(
        "generate-dicom-fixtures",
        help="Generate deterministic safe DICOM CT fixtures.",
    )
    fixture_parser.add_argument("--config", default="config/data.yaml")
    fixture_parser.add_argument("--output-dir", default=None)
    fixture_parser.add_argument("--slice-count", type=int, default=4)
    fixture_parser.add_argument("--malformed", default=None)
    fixture_parser.add_argument("--overwrite", action="store_true")

    discover_parser = subparsers.add_parser("discover-dicom", help="Discover DICOM series.")
    discover_parser.add_argument("input_dir")
    discover_parser.add_argument("--config", default="config/data.yaml")
    discover_parser.add_argument("--json", action="store_true")

    inspect_parser = subparsers.add_parser("inspect-dicom", help="Inspect one DICOM file safely.")
    inspect_parser.add_argument("file_path")
    inspect_parser.add_argument("--config", default="config/data.yaml")
    inspect_parser.add_argument("--include-pixels", action="store_true")
    inspect_parser.add_argument("--json", action="store_true")

    validate_dicom_parser = subparsers.add_parser(
        "validate-dicom", help="Validate one DICOM series directory."
    )
    validate_dicom_parser.add_argument("input_dir")
    validate_dicom_parser.add_argument("--config", default="config/data.yaml")
    validate_dicom_parser.add_argument("--require-pixel-data", action="store_true")
    validate_dicom_parser.add_argument("--json", action="store_true")

    deid_parser = subparsers.add_parser(
        "deidentify-dicom", help="De-identify one DICOM series directory."
    )
    deid_parser.add_argument("input_dir")
    deid_parser.add_argument("--config", default="config/data.yaml")
    deid_parser.add_argument("--output-dir", default=None)
    deid_parser.add_argument("--audit-path", default=None)
    deid_parser.add_argument("--overwrite", action="store_true")

    qc_parser = subparsers.add_parser("quality-check-dicom", help="Run DICOM quality control.")
    qc_parser.add_argument("input_dir")
    qc_parser.add_argument("--config", default="config/data.yaml")
    qc_parser.add_argument("--output-dir", default=None)
    qc_parser.add_argument("--full-pixel-validation", action="store_true")
    qc_parser.add_argument("--json", action="store_true")
    qc_parser.add_argument("--fail-on-warning", action="store_true")
    qc_parser.add_argument("--overwrite", action="store_true")

    report_parser = subparsers.add_parser(
        "quality-report-dicom",
        help="Run DICOM quality control and write reports.",
    )
    report_parser.add_argument("input_dir")
    report_parser.add_argument("--config", default="config/data.yaml")
    report_parser.add_argument("--output-dir", default=None)
    report_parser.add_argument("--full-pixel-validation", action="store_true")
    report_parser.add_argument("--fail-on-warning", action="store_true")
    report_parser.add_argument("--overwrite", action="store_true")

    preprocess_parser = subparsers.add_parser(
        "preprocess-dicom",
        help="Assemble and preprocess one selected DICOM series into a NumPy volume.",
    )
    preprocess_parser.add_argument("input_dir")
    preprocess_parser.add_argument("--config", default="config/preprocessing.yaml")
    preprocess_parser.add_argument("--data-config", default="config/data.yaml")
    preprocess_parser.add_argument("--study-uid", default=None)
    preprocess_parser.add_argument("--series-uid", default=None)
    preprocess_parser.add_argument("--output-dir", default=None)
    preprocess_parser.add_argument("--intensity-profile", default=None)
    preprocess_parser.add_argument(
        "--crop-mode",
        choices=["none", "non_background", "centre", "fixed"],
        default=None,
    )
    preprocess_parser.add_argument("--overwrite", action="store_true")
    preprocess_parser.add_argument("--quality-override", action="store_true")
    preprocess_parser.add_argument("--override-reason", default=None)
    preprocess_parser.add_argument("--json", action="store_true")

    inspect_volume_parser = subparsers.add_parser(
        "inspect-preprocessed-volume",
        help="Inspect an exported preprocessed NumPy volume directory.",
    )
    inspect_volume_parser.add_argument("output_dir")
    inspect_volume_parser.add_argument("--json", action="store_true")

    validate_volume_parser = subparsers.add_parser(
        "validate-preprocessed-volume",
        help="Validate an exported preprocessed NumPy volume directory.",
    )
    validate_volume_parser.add_argument("output_dir")
    validate_volume_parser.add_argument("--json", action="store_true")

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

    if args.command == "generate-dicom-fixtures":
        try:
            dicom_config = load_dicom_ingestion_config(Path(args.config))
            output_dir = (
                Path(args.output_dir)
                if args.output_dir is not None
                else dicom_config.fixture_output_dir
            )
            paths = generate_dicom_fixture_series(
                output_dir,
                slice_count=args.slice_count,
                malformed=args.malformed,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            print(f"DICOM fixture generation failed: {exc}")
            return 1
        print(f"Generated {len(paths)} DICOM fixture files at {output_dir}.")
        return 0

    if args.command == "discover-dicom":
        try:
            dicom_config = load_dicom_ingestion_config(Path(args.config))
            result = discover_dicom_series(
                Path(args.input_dir),
                max_files=dicom_config.max_files,
                max_file_size_bytes=dicom_config.max_file_size_bytes,
            )
        except MedicalImagingPlatformError as exc:
            print(f"DICOM discovery failed: {exc}")
            return 1
        if args.json:
            print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(
                f"Discovered {len(result.series)} DICOM series; "
                f"skipped {len(result.skipped_files)} files."
            )
        return 0

    if args.command == "inspect-dicom":
        try:
            dicom_config = load_dicom_ingestion_config(Path(args.config))
            dataset = load_dicom(
                Path(args.file_path),
                header_only=not args.include_pixels,
                max_file_size_bytes=dicom_config.max_file_size_bytes,
            )
            metadata = extract_metadata(dataset, Path(args.file_path))
        except MedicalImagingPlatformError as exc:
            print(f"DICOM inspection failed: {exc}")
            return 1
        if args.json:
            print(json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(f"DICOM {metadata.sop_instance_uid} modality={metadata.modality}")
        return 0

    if args.command == "validate-dicom":
        try:
            dicom_config = load_dicom_ingestion_config(Path(args.config))
            result = discover_dicom_series(
                Path(args.input_dir),
                max_files=dicom_config.max_files,
                max_file_size_bytes=dicom_config.max_file_size_bytes,
            )
            file_paths = [Path(path) for series in result.series for path in series.files]
            findings = validate_series(
                file_paths,
                accepted_modality=dicom_config.accepted_modality,
                max_file_size_bytes=dicom_config.max_file_size_bytes,
                require_pixel_data=args.require_pixel_data,
            )
        except MedicalImagingPlatformError as exc:
            print(f"DICOM validation failed: {exc}")
            return 1
        if args.json:
            print(
                json.dumps(
                    [finding.model_dump(mode="json") for finding in findings],
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"DICOM validation produced {len(findings)} findings.")
        return 1 if any(finding.severity == "ERROR" for finding in findings) else 0

    if args.command == "deidentify-dicom":
        try:
            dicom_config = load_dicom_ingestion_config(Path(args.config))
            result = discover_dicom_series(
                Path(args.input_dir),
                max_files=dicom_config.max_files,
                max_file_size_bytes=dicom_config.max_file_size_bytes,
            )
            file_paths = [Path(path) for series in result.series for path in series.files]
            deid_settings = dicom_config.deidentification
            policy = default_policy(
                policy_version=str(deid_settings["policy_version"]),
                uid_root=str(deid_settings["uid_root"]),
                patient_id_prefix=str(deid_settings["patient_id_prefix"]),
            )
            output_dir = (
                Path(args.output_dir) if args.output_dir is not None else dicom_config.output_dir
            )
            audit_path = (
                Path(args.audit_path)
                if args.audit_path is not None
                else dicom_config.audit_output_dir / "audit.json"
            )
            audit = deidentify_series(
                file_paths,
                output_dir=output_dir,
                audit_path=audit_path,
                policy=policy,
                overwrite=args.overwrite,
                max_file_size_bytes=dicom_config.max_file_size_bytes,
            )
        except MedicalImagingPlatformError as exc:
            print(f"DICOM de-identification failed: {exc}")
            return 1
        print(
            f"De-identified {audit.file_count} files into {output_dir}; "
            f"audit written to {audit_path}."
        )
        return 0

    if args.command in {"quality-check-dicom", "quality-report-dicom"}:
        try:
            dicom_config = load_dicom_ingestion_config(Path(args.config))
            qc_config = load_quality_control_config(Path(args.config))
            output_dir = (
                Path(args.output_dir) if args.output_dir is not None else qc_config.output_dir
            )
            should_write = args.command == "quality-report-dicom" or args.output_dir is not None
            reports = run_quality_control(
                Path(args.input_dir),
                output_dir=output_dir if should_write else None,
                qc_config=qc_config,
                max_files=dicom_config.max_files,
                max_file_size_bytes=dicom_config.max_file_size_bytes,
                full_pixel_validation=args.full_pixel_validation,
                overwrite=args.overwrite,
            )
        except (MedicalImagingPlatformError, FileExistsError) as exc:
            print(f"DICOM quality control failed: {exc}")
            return 1
        if getattr(args, "json", False):
            print(
                json.dumps(
                    [report.model_dump(mode="json") for report in reports], indent=2, sort_keys=True
                )
            )
        else:
            statuses = ", ".join(report.status for report in reports)
            print(f"DICOM quality control completed for {len(reports)} series: {statuses}.")
        if any(report.status == "REJECTED" for report in reports):
            return 3
        if any(report.status == "FAIL" for report in reports):
            return 2
        if args.fail_on_warning and any(
            report.status == "PASS_WITH_WARNINGS" for report in reports
        ):
            return 2
        return 0

    if args.command == "preprocess-dicom":
        try:
            preprocessing_config = load_preprocessing_config(Path(args.config))
            dicom_config = load_dicom_ingestion_config(Path(args.data_config))
            qc_config = load_quality_control_config(Path(args.data_config))
            output_root = (
                Path(args.output_dir)
                if args.output_dir is not None
                else preprocessing_config.output_directory
            )
            preprocessing_result = preprocess_dicom_series(
                Path(args.input_dir),
                output_root=output_root,
                preprocessing_config=preprocessing_config,
                qc_config=qc_config,
                dicom_max_files=dicom_config.max_files,
                max_file_size_bytes=dicom_config.max_file_size_bytes,
                study_uid=args.study_uid,
                series_uid=args.series_uid,
                intensity_profile=args.intensity_profile,
                crop_mode=args.crop_mode,
                overwrite=args.overwrite,
                quality_override=args.quality_override,
                override_reason=args.override_reason,
            )
        except PreprocessingQualityError as exc:
            print(f"DICOM preprocessing blocked by quality control: {exc}")
            return 2
        except PreprocessingRejectedError as exc:
            print(f"DICOM preprocessing rejected: {exc}")
            return 3
        except PreprocessingOutputError as exc:
            print(f"DICOM preprocessing output validation failed: {exc}")
            return 4
        except (
            ConfigError,
            PreprocessingError,
            MedicalImagingPlatformError,
            FileExistsError,
        ) as exc:
            print(f"DICOM preprocessing failed: {exc}")
            return 1
        if args.json:
            print(
                json.dumps(preprocessing_result.model_dump(mode="json"), indent=2, sort_keys=True)
            )
        else:
            print(
                f"Preprocessed DICOM series into {preprocessing_result.output_paths.output_dir}; "
                f"shape={preprocessing_result.volume_shape}, "
                f"source_qc={preprocessing_result.source_quality_status}."
            )
        return 0

    if args.command == "inspect-preprocessed-volume":
        try:
            summary = inspect_preprocessed_volume(Path(args.output_dir))
        except PreprocessingOutputError as exc:
            print(f"Preprocessed volume inspection failed: {exc}")
            return 4
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(
                f"Preprocessed volume {summary['run_id']} "
                f"shape={summary['shape']} dtype={summary['dtype']}."
            )
        return 0

    if args.command == "validate-preprocessed-volume":
        try:
            validation_result = validate_preprocessed_volume(Path(args.output_dir))
        except PreprocessingOutputError as exc:
            print(f"Preprocessed volume validation failed: {exc}")
            return 4
        if args.json:
            print(json.dumps(validation_result.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(f"Validated preprocessed volume {validation_result.run_id}.")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
