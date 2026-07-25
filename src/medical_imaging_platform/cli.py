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
from medical_imaging_platform.localisation.export import (
    LocalisationOutputError,
    inspect_localisation_output,
    validate_localisation_output,
)
from medical_imaging_platform.localisation.fixtures import generate_localisation_fixture
from medical_imaging_platform.localisation.pipeline import localise_adrenal_regions
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
from medical_imaging_platform.registration.export import (
    RegistrationOutputError,
    inspect_registration_output,
    validate_registration_output,
)
from medical_imaging_platform.registration.fixtures import generate_registration_fixture_pair
from medical_imaging_platform.registration.pipeline import register_preprocessed_volumes
from medical_imaging_platform.synthetic.generator import load_synthetic_config
from medical_imaging_platform.synthetic.io import generate_dataset, validate_dataset
from medical_imaging_platform.utils.config import (
    ConfigError,
    load_api_config,
    load_classification_config,
    load_dicom_ingestion_config,
    load_localisation_config,
    load_longitudinal_config,
    load_preprocessing_config,
    load_quality_control_config,
    load_registration_config,
    load_reviewer_ui_config,
    load_segmentation_config,
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

    registration_fixture_parser = subparsers.add_parser(
        "generate-registration-fixtures",
        help="Generate synthetic preprocessing-compatible registration fixture pair.",
    )
    registration_fixture_parser.add_argument(
        "--output-dir", default="data/processed/registration-fixtures"
    )
    registration_fixture_parser.add_argument("--overwrite", action="store_true")

    register_parser = subparsers.add_parser(
        "register-volumes",
        help="Register an explicit moving preprocessed volume into fixed-volume space.",
    )
    register_parser.add_argument("--fixed", required=True)
    register_parser.add_argument("--moving", required=True)
    register_parser.add_argument("--output-dir", default=None)
    register_parser.add_argument("--config", default="config/registration.yaml")
    register_parser.add_argument(
        "--mode",
        choices=["centre_of_mass", "rigid", "rigid_then_affine"],
        default=None,
    )
    register_parser.add_argument("--fixed-temporal-label", default=None)
    register_parser.add_argument("--moving-temporal-label", default=None)
    register_parser.add_argument("--overwrite", action="store_true")
    register_parser.add_argument("--json", action="store_true")
    register_parser.add_argument("--fail-on-warning", action="store_true")

    inspect_registration_parser = subparsers.add_parser(
        "inspect-registration",
        help="Inspect a registration output directory.",
    )
    inspect_registration_parser.add_argument("output_dir")
    inspect_registration_parser.add_argument("--json", action="store_true")

    validate_registration_parser = subparsers.add_parser(
        "validate-registration",
        help="Validate a registration output directory.",
    )
    validate_registration_parser.add_argument("output_dir")
    validate_registration_parser.add_argument("--json", action="store_true")

    localisation_fixture_parser = subparsers.add_parser(
        "generate-localisation-fixtures",
        help="Generate synthetic preprocessing-compatible localisation fixtures.",
    )
    localisation_fixture_parser.add_argument(
        "--output-dir", default="data/processed/localisation-fixtures"
    )
    localisation_fixture_parser.add_argument("--overwrite", action="store_true")
    localisation_fixture_parser.add_argument("--seed", type=int, default=20260724)
    localisation_fixture_parser.add_argument("--translation", nargs=3, type=int, default=(0, 0, 0))

    localise_parser = subparsers.add_parser(
        "localise-adrenal-regions",
        help="Run deterministic baseline left/right adrenal-region placeholder localisation.",
    )
    localise_parser.add_argument("input_dir")
    localise_parser.add_argument("--config", default="config/localisation.yaml")
    localise_parser.add_argument("--mode", choices=["atlas"], default=None)
    localise_parser.add_argument("--output-dir", default=None)
    localise_parser.add_argument("--left-mask", default=None)
    localise_parser.add_argument("--right-mask", default=None)
    localise_parser.add_argument("--roi-size-voxels", nargs=3, type=int, default=None)
    localise_parser.add_argument("--overwrite", action="store_true")
    localise_parser.add_argument("--json", action="store_true")
    localise_parser.add_argument("--fail-on-warning", action="store_true")

    inspect_localisation_parser = subparsers.add_parser(
        "inspect-localisation",
        help="Inspect a localisation output directory.",
    )
    inspect_localisation_parser.add_argument("output_dir")
    inspect_localisation_parser.add_argument("--json", action="store_true")

    validate_localisation_parser = subparsers.add_parser(
        "validate-localisation",
        help="Validate a localisation output directory.",
    )
    validate_localisation_parser.add_argument("output_dir")
    validate_localisation_parser.add_argument("--json", action="store_true")

    prepare_segmentation_parser = subparsers.add_parser(
        "prepare-segmentation-data",
        help="Prepare synthetic segmentation samples from generated synthetic data.",
    )
    prepare_segmentation_parser.add_argument(
        "--synthetic-dataset-dir", default="data/synthetic/generated"
    )
    prepare_segmentation_parser.add_argument("--config", default="config/segmentation.yaml")
    prepare_segmentation_parser.add_argument("--output-dir", default=None)
    prepare_segmentation_parser.add_argument("--overwrite", action="store_true")
    prepare_segmentation_parser.add_argument("--json", action="store_true")

    train_segmentation_parser = subparsers.add_parser(
        "train-segmentation",
        help="Train the small MONAI 3D U-Net synthetic segmentation baseline.",
    )
    train_segmentation_parser.add_argument("dataset_dir")
    train_segmentation_parser.add_argument("--config", default="config/segmentation.yaml")
    train_segmentation_parser.add_argument("--output-dir", default=None)
    train_segmentation_parser.add_argument("--seed", type=int, default=None)
    train_segmentation_parser.add_argument("--epochs", type=int, default=None)
    train_segmentation_parser.add_argument("--device", default=None)
    train_segmentation_parser.add_argument("--overwrite", action="store_true")
    train_segmentation_parser.add_argument("--json", action="store_true")

    evaluate_segmentation_parser = subparsers.add_parser(
        "evaluate-segmentation",
        help="Validate and inspect a segmentation experiment.",
    )
    evaluate_segmentation_parser.add_argument("experiment_dir")
    evaluate_segmentation_parser.add_argument("--json", action="store_true")

    segment_volume_parser = subparsers.add_parser(
        "segment-volume",
        help="Run segmentation inference for one prepared ROI volume.",
    )
    segment_volume_parser.add_argument("input_volume")
    segment_volume_parser.add_argument("--checkpoint", required=True)
    segment_volume_parser.add_argument("--config", default="config/segmentation.yaml")
    segment_volume_parser.add_argument("--output-dir", required=True)
    segment_volume_parser.add_argument("--threshold", type=float, default=None)
    segment_volume_parser.add_argument("--overwrite", action="store_true")
    segment_volume_parser.add_argument("--json", action="store_true")

    inspect_segmentation_parser = subparsers.add_parser(
        "inspect-segmentation-experiment",
        help="Inspect a segmentation experiment directory.",
    )
    inspect_segmentation_parser.add_argument("experiment_dir")
    inspect_segmentation_parser.add_argument("--json", action="store_true")

    validate_segmentation_parser = subparsers.add_parser(
        "validate-segmentation-experiment",
        help="Validate a segmentation experiment directory.",
    )
    validate_segmentation_parser.add_argument("experiment_dir")
    validate_segmentation_parser.add_argument("--json", action="store_true")

    prepare_classification_parser = subparsers.add_parser(
        "prepare-classification-data",
        help="Prepare synthetic lesion-presence classification ROI samples.",
    )
    prepare_classification_parser.add_argument(
        "--synthetic-dataset-dir", default="data/synthetic/generated"
    )
    prepare_classification_parser.add_argument("--config", default="config/classification.yaml")
    prepare_classification_parser.add_argument("--output-dir", default=None)
    prepare_classification_parser.add_argument("--overwrite", action="store_true")
    prepare_classification_parser.add_argument("--json", action="store_true")

    train_classification_parser = subparsers.add_parser(
        "train-classification",
        help="Train the synthetic lesion-presence classifier.",
    )
    train_classification_parser.add_argument("dataset_dir")
    train_classification_parser.add_argument("--config", default="config/classification.yaml")
    train_classification_parser.add_argument("--output-dir", default=None)
    train_classification_parser.add_argument("--seed", type=int, default=None)
    train_classification_parser.add_argument("--epochs", type=int, default=None)
    train_classification_parser.add_argument("--device", default=None)
    train_classification_parser.add_argument("--overwrite", action="store_true")
    train_classification_parser.add_argument("--json", action="store_true")

    evaluate_classification_parser = subparsers.add_parser(
        "evaluate-classification",
        help="Validate and inspect a classification experiment.",
    )
    evaluate_classification_parser.add_argument("experiment_dir")
    evaluate_classification_parser.add_argument("--json", action="store_true")

    classify_volume_parser = subparsers.add_parser(
        "classify-volume",
        help="Run classification inference for one ROI volume.",
    )
    classify_volume_parser.add_argument("input_volume")
    classify_volume_parser.add_argument("--checkpoint", required=True)
    classify_volume_parser.add_argument("--calibration", required=True)
    classify_volume_parser.add_argument("--threshold-policy", required=True)
    classify_volume_parser.add_argument("--config", default="config/classification.yaml")
    classify_volume_parser.add_argument("--output-dir", required=True)
    classify_volume_parser.add_argument("--overwrite", action="store_true")
    classify_volume_parser.add_argument("--json", action="store_true")

    inspect_classification_parser = subparsers.add_parser(
        "inspect-classification-experiment",
        help="Inspect a classification experiment directory.",
    )
    inspect_classification_parser.add_argument("experiment_dir")
    inspect_classification_parser.add_argument("--json", action="store_true")

    validate_classification_parser = subparsers.add_parser(
        "validate-classification-experiment",
        help="Validate a classification experiment directory.",
    )
    validate_classification_parser.add_argument("experiment_dir")
    validate_classification_parser.add_argument("--json", action="store_true")

    api_config_parser = subparsers.add_parser(
        "validate-api-config",
        help="Validate API configuration.",
    )
    api_config_parser.add_argument("--config", default="config/api.yaml")
    api_config_parser.add_argument("--json", action="store_true")

    api_ready_parser = subparsers.add_parser(
        "inspect-api-readiness",
        help="Inspect API readiness checks without starting a server.",
    )
    api_ready_parser.add_argument("--config", default="config/api.yaml")
    api_ready_parser.add_argument("--json", action="store_true")

    serve_api_parser = subparsers.add_parser(
        "serve-api",
        help="Serve the local governed research API with Uvicorn.",
    )
    serve_api_parser.add_argument("--config", default="config/api.yaml")
    serve_api_parser.add_argument("--host", default=None)
    serve_api_parser.add_argument("--port", type=int, default=None)

    reviewer_ui_config_parser = subparsers.add_parser(
        "validate-reviewer-ui-config",
        help="Validate reviewer UI configuration.",
    )
    reviewer_ui_config_parser.add_argument("--config", default="config/reviewer_ui.yaml")
    reviewer_ui_config_parser.add_argument("--json", action="store_true")

    reviewer_ui_ready_parser = subparsers.add_parser(
        "inspect-reviewer-ui-readiness",
        help="Inspect reviewer UI configuration and API dependency readiness.",
    )
    reviewer_ui_ready_parser.add_argument("--config", default="config/reviewer_ui.yaml")
    reviewer_ui_ready_parser.add_argument("--json", action="store_true")

    serve_reviewer_ui_parser = subparsers.add_parser(
        "serve-reviewer-ui",
        help="Serve the local Streamlit reviewer UI.",
    )
    serve_reviewer_ui_parser.add_argument("--config", default="config/reviewer_ui.yaml")
    serve_reviewer_ui_parser.add_argument("--host", default=None)
    serve_reviewer_ui_parser.add_argument("--port", type=int, default=None)

    longitudinal_parser = subparsers.add_parser(
        "analyse-longitudinal-pair",
        help="Analyse synthetic previous/current lesion masks for engineering change labels.",
    )
    longitudinal_parser.add_argument("--previous-mask", required=True)
    longitudinal_parser.add_argument("--current-mask", required=True)
    longitudinal_parser.add_argument("--previous-spacing", nargs=3, type=float, required=True)
    longitudinal_parser.add_argument("--current-spacing", nargs=3, type=float, required=True)
    longitudinal_parser.add_argument("--case-id", required=True)
    longitudinal_parser.add_argument("--research-subject-id", required=True)
    longitudinal_parser.add_argument("--side", choices=["left", "right"], required=True)
    longitudinal_parser.add_argument("--previous-timepoint", default="previous")
    longitudinal_parser.add_argument("--current-timepoint", default="current")
    longitudinal_parser.add_argument("--registration-run-id", default=None)
    longitudinal_parser.add_argument("--localisation-run-id", action="append", default=[])
    longitudinal_parser.add_argument("--segmentation-run-id", action="append", default=[])
    longitudinal_parser.add_argument("--classification-run-id", action="append", default=[])
    longitudinal_parser.add_argument("--registration-status", default="PASS")
    longitudinal_parser.add_argument("--segmentation-status", default="PASS")
    longitudinal_parser.add_argument("--localisation-status", default="PASS")
    longitudinal_parser.add_argument("--classification-status", default="PASS")
    longitudinal_parser.add_argument("--classification-abstention-status", default="NOT_ABSTAINED")
    longitudinal_parser.add_argument("--upstream-status-json", default=None)
    longitudinal_parser.add_argument("--config", default="config/longitudinal.yaml")
    longitudinal_parser.add_argument("--output-dir", default=None)
    longitudinal_parser.add_argument("--overwrite", action="store_true")
    longitudinal_parser.add_argument("--json", action="store_true")

    inspect_longitudinal_parser = subparsers.add_parser(
        "inspect-longitudinal-analysis",
        help="Inspect a longitudinal analysis output directory.",
    )
    inspect_longitudinal_parser.add_argument("analysis_dir")
    inspect_longitudinal_parser.add_argument("--json", action="store_true")

    validate_longitudinal_parser = subparsers.add_parser(
        "validate-longitudinal-analysis",
        help="Validate a longitudinal analysis output directory.",
    )
    validate_longitudinal_parser.add_argument("analysis_dir")
    validate_longitudinal_parser.add_argument("--json", action="store_true")

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

    if args.command == "generate-registration-fixtures":
        try:
            fixed_dir, moving_dir = generate_registration_fixture_pair(
                Path(args.output_dir), overwrite=args.overwrite
            )
        except Exception as exc:
            print(f"Registration fixture generation failed: {exc}")
            return 1
        print(f"Generated registration fixtures fixed={fixed_dir} moving={moving_dir}.")
        return 0

    if args.command == "register-volumes":
        try:
            registration_config = load_registration_config(Path(args.config))
            output_root = (
                Path(args.output_dir)
                if args.output_dir is not None
                else registration_config.output_directory
            )
            registration_result = register_preprocessed_volumes(
                Path(args.fixed),
                Path(args.moving),
                output_root=output_root,
                config=registration_config,
                mode=args.mode,
                fixed_temporal_label=args.fixed_temporal_label,
                moving_temporal_label=args.moving_temporal_label,
                overwrite=args.overwrite,
            )
        except RegistrationOutputError as exc:
            print(f"Registration output validation failed: {exc}")
            return 4
        except ValueError as exc:
            print(f"Registration rejected: {exc}")
            return 3
        except (ConfigError, MedicalImagingPlatformError, FileExistsError) as exc:
            print(f"Registration failed: {exc}")
            return 1
        if args.json:
            print(json.dumps(registration_result.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(
                f"Registered moving volume into fixed space at "
                f"{registration_result.output_paths.output_dir}; "
                f"status={registration_result.status}."
            )
        if registration_result.status == "REJECTED":
            return 3
        if registration_result.status == "FAIL":
            return 2
        if args.fail_on_warning and registration_result.status == "PASS_WITH_WARNINGS":
            return 2
        return 0

    if args.command == "inspect-registration":
        try:
            summary = inspect_registration_output(Path(args.output_dir))
        except RegistrationOutputError as exc:
            print(f"Registration inspection failed: {exc}")
            return 4
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(
                f"Registration {summary['registration_run_id']} "
                f"status={summary['status']} shape={summary['shape']}."
            )
        return 0

    if args.command == "validate-registration":
        try:
            registration_result = validate_registration_output(Path(args.output_dir))
        except RegistrationOutputError as exc:
            print(f"Registration validation failed: {exc}")
            return 4
        if args.json:
            print(json.dumps(registration_result.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(f"Validated registration {registration_result.registration_run_id}.")
        return 0

    if args.command == "generate-localisation-fixtures":
        try:
            fixture_dir = generate_localisation_fixture(
                Path(args.output_dir),
                translation=tuple(args.translation),
                random_seed=args.seed,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            print(f"Localisation fixture generation failed: {exc}")
            return 1
        print(f"Generated localisation fixture at {fixture_dir}.")
        return 0

    if args.command == "localise-adrenal-regions":
        try:
            localisation_config = load_localisation_config(Path(args.config))
            if args.roi_size_voxels is not None:
                localisation_config = localisation_config.model_copy(
                    update={
                        "roi_size_voxels": tuple(args.roi_size_voxels),
                        "roi_size_mm": None,
                    }
                )
            output_root = (
                Path(args.output_dir)
                if args.output_dir is not None
                else localisation_config.output_directory
            )
            localisation_result = localise_adrenal_regions(
                Path(args.input_dir),
                output_root=output_root,
                config=localisation_config,
                mode=args.mode,
                left_mask_path=Path(args.left_mask) if args.left_mask is not None else None,
                right_mask_path=Path(args.right_mask) if args.right_mask is not None else None,
                overwrite=args.overwrite,
            )
        except LocalisationOutputError as exc:
            print(f"Localisation output validation failed: {exc}")
            return 4
        except ValueError as exc:
            print(f"Localisation rejected: {exc}")
            return 3
        except (ConfigError, MedicalImagingPlatformError, FileExistsError) as exc:
            print(f"Localisation failed: {exc}")
            return 1
        if args.json:
            print(json.dumps(localisation_result.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(
                f"Localised adrenal-region placeholders at "
                f"{localisation_result.output_paths.output_dir}; "
                f"status={localisation_result.overall_status}."
            )
        if localisation_result.overall_status == "REJECTED":
            return 3
        if localisation_result.overall_status == "FAILED":
            return 2
        if args.fail_on_warning and localisation_result.overall_status == "LOCALISED_WITH_WARNINGS":
            return 2
        return 0

    if args.command == "inspect-localisation":
        try:
            summary = inspect_localisation_output(Path(args.output_dir))
        except LocalisationOutputError as exc:
            print(f"Localisation inspection failed: {exc}")
            return 4
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(
                f"Localisation {summary['localisation_run_id']} "
                f"status={summary['status']} source={summary['source_run_id']}."
            )
        return 0

    if args.command == "validate-localisation":
        try:
            localisation_result = validate_localisation_output(Path(args.output_dir))
        except LocalisationOutputError as exc:
            print(f"Localisation validation failed: {exc}")
            return 4
        if args.json:
            print(json.dumps(localisation_result.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(f"Validated localisation {localisation_result.localisation_run_id}.")
        return 0

    if args.command == "prepare-segmentation-data":
        from medical_imaging_platform.segmentation.dataset import (
            SegmentationDataError,
            prepare_segmentation_dataset,
        )

        try:
            segmentation_config = load_segmentation_config(Path(args.config))
            output_root = (
                Path(args.output_dir)
                if args.output_dir is not None
                else segmentation_config.dataset_output_directory
            )
            segmentation_manifest = prepare_segmentation_dataset(
                Path(args.synthetic_dataset_dir),
                output_root=output_root,
                config=segmentation_config,
                overwrite=args.overwrite,
            )
        except (
            ConfigError,
            SegmentationDataError,
            MedicalImagingPlatformError,
            FileExistsError,
        ) as exc:
            print(f"Segmentation dataset preparation failed: {exc}")
            return 1
        if args.json:
            print(
                json.dumps(segmentation_manifest.model_dump(mode="json"), indent=2, sort_keys=True)
            )
        else:
            print(
                f"Prepared segmentation dataset {segmentation_manifest.dataset_id} "
                f"with {len(segmentation_manifest.samples)} samples."
            )
        return 0

    if args.command == "train-segmentation":
        from medical_imaging_platform.segmentation.export import SegmentationOutputError
        from medical_imaging_platform.segmentation.trainer import train_segmentation_experiment

        try:
            segmentation_config = load_segmentation_config(Path(args.config))
            output_root = (
                Path(args.output_dir)
                if args.output_dir is not None
                else segmentation_config.output_directory
            )
            segmentation_payload = train_segmentation_experiment(
                Path(args.dataset_dir),
                output_root=output_root,
                config=segmentation_config,
                seed=args.seed,
                epochs=args.epochs,
                device_name=args.device,
                overwrite=args.overwrite,
            )
        except SegmentationOutputError as exc:
            print(f"Segmentation experiment output validation failed: {exc}")
            return 4
        except ValueError as exc:
            print(f"Segmentation experiment rejected: {exc}")
            return 3
        except (ConfigError, MedicalImagingPlatformError, FileExistsError) as exc:
            print(f"Segmentation training failed: {exc}")
            return 1
        if args.json:
            print(json.dumps(segmentation_payload, indent=2, sort_keys=True))
        else:
            print(
                f"Trained segmentation experiment {segmentation_payload['experiment_id']} "
                f"status={segmentation_payload['status']}."
            )
        if segmentation_payload["status"] == "REJECTED":
            return 3
        if segmentation_payload["status"] == "FAIL":
            return 2
        return 0

    if args.command in {
        "evaluate-segmentation",
        "inspect-segmentation-experiment",
        "validate-segmentation-experiment",
    }:
        from medical_imaging_platform.segmentation.export import (
            SegmentationOutputError,
            inspect_segmentation_experiment,
            validate_segmentation_experiment,
        )

        try:
            if args.command == "validate-segmentation-experiment":
                segmentation_summary = validate_segmentation_experiment(Path(args.experiment_dir))
            else:
                segmentation_summary = inspect_segmentation_experiment(Path(args.experiment_dir))
        except SegmentationOutputError as exc:
            print(f"Segmentation experiment validation failed: {exc}")
            return 4
        if args.json:
            print(json.dumps(segmentation_summary, indent=2, sort_keys=True))
        else:
            print(
                f"Segmentation experiment {segmentation_summary['experiment_id']} "
                f"status={segmentation_summary['status']}."
            )
        return 0

    if args.command == "segment-volume":
        from medical_imaging_platform.segmentation.inference import segment_volume

        try:
            segmentation_config = load_segmentation_config(Path(args.config))
            segmentation_metadata = segment_volume(
                Path(args.input_volume),
                checkpoint_path=Path(args.checkpoint),
                output_dir=Path(args.output_dir),
                config=segmentation_config,
                threshold=args.threshold,
                overwrite=args.overwrite,
            )
        except ValueError as exc:
            print(f"Segmentation inference rejected: {exc}")
            return 3
        except (ConfigError, MedicalImagingPlatformError, FileExistsError) as exc:
            print(f"Segmentation inference failed: {exc}")
            return 1
        if args.json:
            print(json.dumps(segmentation_metadata, indent=2, sort_keys=True))
        else:
            print(f"Wrote segmentation inference outputs to {args.output_dir}.")
        return 0

    if args.command == "prepare-classification-data":
        from medical_imaging_platform.classification.dataset import (
            ClassificationDataError,
            prepare_classification_dataset,
        )

        try:
            classification_config = load_classification_config(Path(args.config))
            output_root = (
                Path(args.output_dir)
                if args.output_dir is not None
                else classification_config.dataset_output_directory
            )
            classification_manifest = prepare_classification_dataset(
                Path(args.synthetic_dataset_dir),
                output_root=output_root,
                config=classification_config,
                overwrite=args.overwrite,
            )
        except (
            ConfigError,
            ClassificationDataError,
            MedicalImagingPlatformError,
            FileExistsError,
        ) as exc:
            print(f"Classification dataset preparation failed: {exc}")
            return 1
        if args.json:
            print(
                json.dumps(
                    classification_manifest.model_dump(mode="json"), indent=2, sort_keys=True
                )
            )
        else:
            print(
                f"Prepared classification dataset {classification_manifest.dataset_id} "
                f"with {len(classification_manifest.samples)} samples."
            )
        return 0

    if args.command == "train-classification":
        from medical_imaging_platform.classification.trainer import train_classification_experiment

        try:
            classification_config = load_classification_config(Path(args.config))
            output_root = (
                Path(args.output_dir)
                if args.output_dir is not None
                else classification_config.output_directory
            )
            classification_payload = train_classification_experiment(
                Path(args.dataset_dir),
                output_root=output_root,
                config=classification_config,
                seed=args.seed,
                epochs=args.epochs,
                device_name=args.device,
                overwrite=args.overwrite,
            )
        except ValueError as exc:
            print(f"Classification experiment rejected: {exc}")
            return 3
        except (ConfigError, MedicalImagingPlatformError, FileExistsError) as exc:
            print(f"Classification training failed: {exc}")
            return 1
        if args.json:
            print(json.dumps(classification_payload, indent=2, sort_keys=True))
        else:
            print(
                f"Trained classification experiment {classification_payload['experiment_id']} "
                f"status={classification_payload['status']}."
            )
        if classification_payload["status"] == "REJECTED":
            return 3
        if classification_payload["status"] == "FAIL":
            return 2
        return 0

    if args.command in {
        "evaluate-classification",
        "inspect-classification-experiment",
        "validate-classification-experiment",
    }:
        from medical_imaging_platform.classification.export import (
            ClassificationOutputError,
            inspect_classification_experiment,
            validate_classification_experiment,
        )

        try:
            if args.command == "validate-classification-experiment":
                classification_summary = validate_classification_experiment(
                    Path(args.experiment_dir)
                )
            else:
                classification_summary = inspect_classification_experiment(
                    Path(args.experiment_dir)
                )
        except ClassificationOutputError as exc:
            print(f"Classification experiment validation failed: {exc}")
            return 4
        if args.json:
            print(json.dumps(classification_summary, indent=2, sort_keys=True))
        else:
            print(
                f"Classification experiment {classification_summary['experiment_id']} "
                f"status={classification_summary['status']}."
            )
        return 0

    if args.command == "classify-volume":
        from medical_imaging_platform.classification.inference import classify_volume

        try:
            classification_config = load_classification_config(Path(args.config))
            classification_metadata = classify_volume(
                Path(args.input_volume),
                checkpoint_path=Path(args.checkpoint),
                calibration_path=Path(args.calibration),
                threshold_policy_path=Path(args.threshold_policy),
                output_dir=Path(args.output_dir),
                config=classification_config,
                overwrite=args.overwrite,
            )
        except ValueError as exc:
            print(f"Classification inference rejected: {exc}")
            return 3
        except (ConfigError, MedicalImagingPlatformError, FileExistsError) as exc:
            print(f"Classification inference failed: {exc}")
            return 1
        if args.json:
            print(json.dumps(classification_metadata, indent=2, sort_keys=True))
        else:
            print(f"Wrote classification inference outputs to {args.output_dir}.")
        return 0

    if args.command == "validate-api-config":
        try:
            api_config = load_api_config(Path(args.config))
        except ConfigError as exc:
            print(f"API configuration validation failed: {exc}")
            return 1
        payload = api_config.model_dump(mode="json")
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Validated API configuration {api_config.policy_version}.")
        return 0

    if args.command == "inspect-api-readiness":
        from medical_imaging_platform.api.routes.health import readiness_findings

        try:
            api_config = load_api_config(Path(args.config))
            api_findings = readiness_findings(api_config)
        except ConfigError as exc:
            print(f"API readiness inspection failed: {exc}")
            return 1
        payload = {
            "status": "ready" if all(item["passed"] for item in api_findings) else "not_ready",
            "quality_findings": api_findings,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"API readiness status={payload['status']}.")
        return 0 if payload["status"] == "ready" else 3

    if args.command == "serve-api":
        try:
            api_config = load_api_config(Path(args.config))
            host = args.host or api_config.host
            port = args.port or api_config.port
            if host == "0.0.0.0" and not api_config.allow_external_bind:  # nosec B104
                print("API serve rejected: 0.0.0.0 requires allow_external_bind=true.")
                return 3
        except ConfigError as exc:
            print(f"API serve failed: {exc}")
            return 1
        import uvicorn

        print(f"Starting API {api_config.policy_version} on {host}:{port}.")
        uvicorn.run(
            "medical_imaging_platform.api.app:create_app",
            factory=True,
            host=host,
            port=port,
            reload=False,
        )
        return 0

    if args.command == "validate-reviewer-ui-config":
        try:
            reviewer_config = load_reviewer_ui_config(Path(args.config))
        except ConfigError as exc:
            print(f"Reviewer UI configuration validation failed: {exc}")
            return 1
        payload = reviewer_config.model_dump(mode="json")
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Validated reviewer UI configuration {reviewer_config.policy_version}.")
        return 0

    if args.command == "inspect-reviewer-ui-readiness":
        from medical_imaging_platform.reviewer_ui.api_client import ReviewerAPIClient
        from medical_imaging_platform.reviewer_ui.models import ReviewerAPIError

        try:
            reviewer_config = load_reviewer_ui_config(Path(args.config))
        except ConfigError as exc:
            print(f"Reviewer UI readiness inspection failed: {exc}")
            return 1
        ui_findings: list[dict[str, object]] = [
            {
                "rule_id": "UI-QC-CONFIG-001",
                "passed": True,
                "message": "Reviewer UI configuration loaded.",
            }
        ]
        client = ReviewerAPIClient(reviewer_config)
        try:
            health = client.health()
            ready = client.ready()
            api_message = (
                f"API dependency status health={health.get('status')} "
                f"readiness={ready.get('status')}."
            )
            ui_findings.append(
                {
                    "rule_id": "UI-QC-API-001",
                    "passed": health.get("status") == "healthy" and ready.get("status") == "ready",
                    "message": api_message,
                }
            )
        except ReviewerAPIError as exc:
            ui_findings.append(
                {
                    "rule_id": "UI-QC-API-001",
                    "passed": False,
                    "message": exc.message,
                    "error_code": exc.error_code,
                }
            )
        payload = {
            "status": "ready" if all(item["passed"] for item in ui_findings) else "not_ready",
            "quality_findings": ui_findings,
            "api_base_url": reviewer_config.api_base_url,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Reviewer UI readiness status={payload['status']}.")
        return 0 if payload["status"] == "ready" else 3

    if args.command == "serve-reviewer-ui":
        try:
            reviewer_config = load_reviewer_ui_config(Path(args.config))
            host = args.host or reviewer_config.host
            port = args.port or reviewer_config.port
            if host == "0.0.0.0" and not reviewer_config.allow_remote_bind:  # nosec B104
                print("Reviewer UI serve rejected: 0.0.0.0 requires allow_remote_bind=true.")
                return 3
        except ConfigError as exc:
            print(f"Reviewer UI serve failed: {exc}")
            return 1
        import sys

        from streamlit.web import cli as streamlit_cli

        app_path = Path(__file__).parent / "reviewer_ui" / "app.py"
        print(
            f"Starting reviewer UI {reviewer_config.policy_version} on {host}:{port}; "
            f"API dependency={reviewer_config.api_base_url}."
        )
        sys.argv = [
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            host,
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]
        streamlit_cli.main()
        return 0

    if args.command == "analyse-longitudinal-pair":
        from medical_imaging_platform.longitudinal.pipeline import (
            LongitudinalAnalysisError,
            analyse_longitudinal_pair,
            load_upstream_statuses,
        )

        try:
            longitudinal_config = load_longitudinal_config(Path(args.config))
            output_root = (
                Path(args.output_dir)
                if args.output_dir is not None
                else longitudinal_config.output_directory
            )
            upstream_statuses = load_upstream_statuses(
                Path(args.upstream_status_json) if args.upstream_status_json else None
            ) or {
                "registration": args.registration_status,
                "segmentation": args.segmentation_status,
                "localisation": args.localisation_status,
                "classification": args.classification_status,
                "classification_abstention": args.classification_abstention_status,
            }
            longitudinal_payload = analyse_longitudinal_pair(
                previous_mask_path=Path(args.previous_mask),
                current_mask_path=Path(args.current_mask),
                previous_spacing_mm=tuple(args.previous_spacing),
                current_spacing_mm=tuple(args.current_spacing),
                case_id=args.case_id,
                research_subject_id=args.research_subject_id,
                side=args.side,
                previous_timepoint=args.previous_timepoint,
                current_timepoint=args.current_timepoint,
                output_root=output_root,
                config=longitudinal_config,
                registration_run_id=args.registration_run_id,
                localisation_run_ids=args.localisation_run_id,
                segmentation_run_ids=args.segmentation_run_id,
                classification_run_ids=args.classification_run_id,
                upstream_quality_statuses=upstream_statuses,
                overwrite=args.overwrite,
            )
        except (ConfigError, LongitudinalAnalysisError, ValueError) as exc:
            print(f"Longitudinal analysis rejected: {exc}")
            return 3
        except (MedicalImagingPlatformError, FileExistsError) as exc:
            print(f"Longitudinal analysis failed: {exc}")
            return 1
        if args.json:
            print(json.dumps(longitudinal_payload, indent=2, sort_keys=True))
        else:
            print(
                f"Analysed longitudinal pair {longitudinal_payload['analysis_id']} "
                f"status={longitudinal_payload['status']}."
            )
        if longitudinal_payload["status"] == "REJECTED":
            return 3
        if longitudinal_payload["status"] == "FAIL":
            return 2
        return 0

    if args.command in {"inspect-longitudinal-analysis", "validate-longitudinal-analysis"}:
        from medical_imaging_platform.longitudinal.export import (
            LongitudinalOutputError,
            inspect_longitudinal_analysis,
            validate_longitudinal_analysis,
        )

        try:
            if args.command == "validate-longitudinal-analysis":
                longitudinal_summary = validate_longitudinal_analysis(Path(args.analysis_dir))
            else:
                longitudinal_summary = inspect_longitudinal_analysis(Path(args.analysis_dir))
        except LongitudinalOutputError as exc:
            print(f"Longitudinal analysis validation failed: {exc}")
            return 4
        if args.json:
            print(json.dumps(longitudinal_summary, indent=2, sort_keys=True))
        else:
            print(
                f"Longitudinal analysis {longitudinal_summary['analysis_id']} "
                f"status={longitudinal_summary['status']}."
            )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
