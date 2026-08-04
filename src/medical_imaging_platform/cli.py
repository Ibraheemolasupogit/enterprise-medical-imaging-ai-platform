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
    load_container_config,
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

    container_config_parser = subparsers.add_parser(
        "validate-container-config",
        help="Validate local container release-assurance configuration.",
    )
    container_config_parser.add_argument("--config", default="config/container.yaml")
    container_config_parser.add_argument("--json", action="store_true")

    container_security_parser = subparsers.add_parser(
        "inspect-container-security",
        help="Inspect Dockerfiles, Compose and build-context security controls.",
    )
    container_security_parser.add_argument("--config", default="config/container.yaml")
    container_security_parser.add_argument("--json", action="store_true")

    release_manifest_parser = subparsers.add_parser(
        "build-release-manifest",
        help="Build ignored local release evidence for container assurance.",
    )
    release_manifest_parser.add_argument("--config", default="config/container.yaml")
    release_manifest_parser.add_argument("--overwrite", action="store_true")
    release_manifest_parser.add_argument("--no-smoke-execute", action="store_true")
    release_manifest_parser.add_argument("--json", action="store_true")

    validate_release_parser = subparsers.add_parser(
        "validate-release-evidence",
        help="Validate generated local release evidence and checksums.",
    )
    validate_release_parser.add_argument("--release-dir", default=None)
    validate_release_parser.add_argument("--json", action="store_true")

    smoke_parser = subparsers.add_parser(
        "run-container-smoke-tests",
        help="Run local Docker Compose smoke tests without publishing images.",
    )
    smoke_parser.add_argument("--config", default="config/container.yaml")
    smoke_parser.add_argument("--no-execute", action="store_true")
    smoke_parser.add_argument("--json", action="store_true")

    secrets_parser = subparsers.add_parser(
        "scan-release-secrets",
        help="Run optional local secret scanning for release assurance.",
    )
    secrets_parser.add_argument("--config", default="config/container.yaml")
    secrets_parser.add_argument("--json", action="store_true")

    dependencies_parser = subparsers.add_parser(
        "scan-release-dependencies",
        help="Run optional local dependency vulnerability scanning.",
    )
    dependencies_parser.add_argument("--config", default="config/container.yaml")
    dependencies_parser.add_argument("--json", action="store_true")

    sbom_parser = subparsers.add_parser(
        "generate-release-sbom",
        help="Generate local SBOM evidence for API and reviewer UI build contexts.",
    )
    sbom_parser.add_argument("--config", default="config/container.yaml")
    sbom_parser.add_argument("--output-dir", default=None)
    sbom_parser.add_argument("--json", action="store_true")

    image_scan_parser = subparsers.add_parser(
        "scan-release-images",
        help="Run optional local image vulnerability scanning.",
    )
    image_scan_parser.add_argument("--config", default="config/container.yaml")
    image_scan_parser.add_argument("--json", action="store_true")

    register_model_parser = subparsers.add_parser(
        "register-model",
        help="Register deterministic synthetic model versions in the local governed registry.",
    )
    register_model_parser.add_argument(
        "--model-type", choices=["segmentation", "classification", "all"], default="all"
    )
    register_model_parser.add_argument("--registry-path", default=None)
    register_model_parser.add_argument("--json", action="store_true")

    list_models_parser = subparsers.add_parser(
        "list-models",
        help="List local governed model registry records.",
    )
    list_models_parser.add_argument("--registry-path", default=None)
    list_models_parser.add_argument("--json", action="store_true")

    approve_model_parser = subparsers.add_parser(
        "approve-model",
        help="Explicitly approve a registered model version with human governance metadata.",
    )
    approve_model_parser.add_argument("--model-name", required=True)
    approve_model_parser.add_argument("--version", required=True)
    approve_model_parser.add_argument("--approved-by", required=True)
    approve_model_parser.add_argument("--approval-ticket", required=True)
    approve_model_parser.add_argument("--rationale", required=True)
    approve_model_parser.add_argument("--approval-timestamp", default="2026-01-01T00:05:00Z")
    approve_model_parser.add_argument("--registry-path", default=None)
    approve_model_parser.add_argument("--json", action="store_true")

    monitoring_baseline_parser = subparsers.add_parser(
        "build-monitoring-baseline",
        help="Build deterministic synthetic monitoring baseline evidence.",
    )
    monitoring_baseline_parser.add_argument("--json", action="store_true")

    run_monitoring_parser = subparsers.add_parser(
        "run-monitoring",
        help="Run deterministic synthetic monitoring against the stored baseline.",
    )
    run_monitoring_parser.add_argument("--json", action="store_true")

    simulate_drift_parser = subparsers.add_parser(
        "simulate-monitoring-drift",
        help="Run deterministic synthetic drift simulation against the stored baseline.",
    )
    simulate_drift_parser.add_argument("--json", action="store_true")

    validate_monitoring_parser = subparsers.add_parser(
        "validate-monitoring-evidence",
        help="Validate generated registry, monitoring, and audit evidence.",
    )
    validate_monitoring_parser.add_argument("--json", action="store_true")

    audit_evidence_parser = subparsers.add_parser(
        "build-audit-evidence",
        help="Build deterministic append-only synthetic audit evidence.",
    )
    audit_evidence_parser.add_argument("--json", action="store_true")

    kubernetes_commands = {
        "validate-helm": "Validate Helm chart structure and optional helm lint.",
        "render-kubernetes": "Render deterministic Kubernetes manifests for local evidence.",
        "validate-kubernetes-policy": (
            "Validate rendered manifests against secure Kubernetes policy."
        ),
        "deploy-local-kubernetes": "Record local Kubernetes deployment availability.",
        "kubernetes-smoke": "Run or record local Kubernetes smoke-test availability.",
        "build-kubernetes-evidence": "Build deterministic Kubernetes deployment evidence.",
        "validate-kubernetes-evidence": "Validate generated Kubernetes deployment evidence.",
        "clean-local-kubernetes": "Record local Kubernetes cleanup evidence.",
    }
    for command_name, help_text in kubernetes_commands.items():
        command_parser = subparsers.add_parser(command_name, help=help_text)
        command_parser.add_argument("--json", action="store_true")

    aws_commands = {
        "terraform-fmt-check": "Run terraform fmt -check for the AWS IaC tree.",
        "terraform-init": "Run terraform init with the backend disabled.",
        "terraform-validate": "Run terraform validate for the AWS IaC tree.",
        "validate-aws-policy": "Run deterministic AWS/Terraform policy checks.",
        "scan-terraform": "Run optional Terraform security scanners when available.",
        "build-aws-evidence": "Build deterministic AWS architecture evidence.",
        "validate-aws-evidence": "Validate generated AWS architecture evidence.",
        "clean-terraform": "Remove local Terraform cache artefacts.",
        "aws-plan": "Optionally run terraform plan without applying resources.",
    }
    for command_name, help_text in aws_commands.items():
        command_parser = subparsers.add_parser(command_name, help=help_text)
        command_parser.add_argument("--json", action="store_true")

    operations_commands = {
        "validate-observability": "Validate operations observability controls.",
        "build-observability-evidence": "Build deterministic observability evidence.",
        "evaluate-slos": "Evaluate demonstrator SLOs and error budgets.",
        "simulate-incidents": "Simulate deterministic incident scenarios.",
        "build-incident-evidence": "Build incident lifecycle evidence.",
        "validate-runbooks": "Validate operations runbooks.",
        "simulate-rollback": "Simulate rollback planning evidence.",
        "validate-recovery": "Validate recovery evidence.",
        "build-operations-evidence": "Build complete operations evidence.",
        "validate-operations-evidence": "Validate generated operations evidence.",
    }
    for command_name, help_text in operations_commands.items():
        command_parser = subparsers.add_parser(command_name, help=help_text)
        command_parser.add_argument("--json", action="store_true")

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

    if args.command == "validate-container-config":
        try:
            container_config = load_container_config(Path(args.config))
        except ConfigError as exc:
            print(f"Container configuration validation failed: {exc}")
            return 1
        payload = container_config.model_dump(mode="json")
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Validated container configuration {container_config.policy_version}.")
        return 0

    if args.command == "inspect-container-security":
        from medical_imaging_platform.release.compose import inspect_compose
        from medical_imaging_platform.release.dependencies import (
            inspect_container_dependency_policy,
        )
        from medical_imaging_platform.release.dockerfiles import inspect_required_dockerfiles
        from medical_imaging_platform.release.evidence import write_tool_evidence
        from medical_imaging_platform.release.image_policy import inspect_dockerignore
        from medical_imaging_platform.release.scanners import scanner_inventory

        try:
            container_config = load_container_config(Path(args.config))
        except ConfigError as exc:
            print(f"Container security inspection failed: {exc}")
            return 1
        checks = [
            *inspect_required_dockerfiles(container_config),
            *inspect_container_dependency_policy(container_config),
            *inspect_compose(container_config),
            *inspect_dockerignore(),
        ]
        tools = scanner_inventory(timeout_seconds=15)
        hadolint = next((tool for tool in tools if tool.tool == "hadolint"), None)
        if hadolint is not None:
            write_tool_evidence(
                container_config,
                "hadolint",
                hadolint.model_copy(
                    update={
                        "details": {
                            **hadolint.details,
                            "optional_reason": (
                                "Internal Dockerfile validation is the mandatory lint gate; "
                                "Hadolint is optional advisory evidence."
                            ),
                        }
                    }
                ),
            )
        payload = {
            "status": "PASS" if all(check.status == "PASS" for check in checks) else "FAIL",
            "checks": [check.model_dump(mode="json") for check in checks],
            "tools": [tool.model_dump(mode="json") for tool in tools],
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"Container security inspection status={payload['status']} "
                f"checks={len(checks)} tools={len(tools)}."
            )
        return 0 if payload["status"] == "PASS" else 2

    if args.command == "build-release-manifest":
        from medical_imaging_platform.release.compose import inspect_compose
        from medical_imaging_platform.release.dependencies import (
            inspect_container_dependency_policy,
        )
        from medical_imaging_platform.release.dockerfiles import inspect_required_dockerfiles
        from medical_imaging_platform.release.evidence import (
            load_canonical_tool_evidence,
            load_smoke_evidence,
        )
        from medical_imaging_platform.release.export import export_release_evidence
        from medical_imaging_platform.release.image_policy import inspect_dockerignore
        from medical_imaging_platform.release.manifest import build_release_manifest

        try:
            container_config = load_container_config(Path(args.config))
            checks = [
                *inspect_required_dockerfiles(container_config),
                *inspect_container_dependency_policy(container_config),
                *inspect_compose(container_config),
                *inspect_dockerignore(),
            ]
            scanner_results = load_canonical_tool_evidence(container_config)
            smoke_result = load_smoke_evidence(container_config)
            release_manifest = build_release_manifest(
                container_config,
                checks,
                scanner_results,
                smoke_result,
            )
            output_dir = export_release_evidence(
                container_config,
                release_manifest,
                overwrite=args.overwrite,
            )
        except (ConfigError, FileExistsError, OSError, ValueError) as exc:
            print(f"Release manifest build failed: {exc}")
            return 1
        payload = {
            "release_dir": output_dir.as_posix(),
            "manifest": release_manifest.model_dump(mode="json"),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Built release evidence at {output_dir}.")
        return 0

    if args.command == "validate-release-evidence":
        from medical_imaging_platform.release.export import validate_release_evidence

        try:
            release_dir = _resolve_release_dir(Path(args.release_dir) if args.release_dir else None)
            checks = validate_release_evidence(release_dir)
        except (OSError, ValueError) as exc:
            print(f"Release evidence validation failed: {exc}")
            return 1
        manifest_status = checks[-1].status if checks else "ERROR"
        payload = {
            "release_dir": release_dir.as_posix(),
            "status": manifest_status
            if all(check.status == "PASS" for check in checks[:-1])
            else "FAIL",
            "checks": [check.model_dump(mode="json") for check in checks],
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Release evidence validation status={payload['status']} at {release_dir}.")
        return 0 if payload["status"] == "PASS" else 2

    if args.command == "run-container-smoke-tests":
        from medical_imaging_platform.release.evidence import write_smoke_evidence
        from medical_imaging_platform.release.smoke import run_container_smoke_tests

        try:
            container_config = load_container_config(Path(args.config))
            smoke_result = run_container_smoke_tests(
                container_config,
                execute=not args.no_execute,
            )
        except ConfigError as exc:
            print(f"Container smoke tests failed: {exc}")
            return 1
        payload = smoke_result.model_dump(mode="json")
        write_smoke_evidence(container_config, smoke_result)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Container smoke-test status={smoke_result.status}.")
        if smoke_result.status == "UNAVAILABLE":
            return 7
        return 0 if smoke_result.status in {"PASS", "SKIPPED"} else 2

    if args.command == "scan-release-secrets":
        from medical_imaging_platform.release.evidence import write_tool_evidence
        from medical_imaging_platform.release.scanners import scan_repository_secrets

        try:
            container_config = load_container_config(Path(args.config))
        except ConfigError as exc:
            print(f"Secret scan failed: {exc}")
            return 1
        secret_scan_result = scan_repository_secrets(container_config.scanner_timeout_seconds)
        write_tool_evidence(container_config, "gitleaks", secret_scan_result)
        if args.json:
            print(json.dumps(secret_scan_result.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(f"Secret scan status={secret_scan_result.status} tool={secret_scan_result.tool}.")
            if secret_scan_result.output:
                print(secret_scan_result.output)
        return 0 if secret_scan_result.status in {"PASS", "UNAVAILABLE"} else 2

    if args.command == "scan-release-dependencies":
        from medical_imaging_platform.release.evidence import write_tool_evidence
        from medical_imaging_platform.release.scanners import scan_dependencies, severity_gate

        try:
            container_config = load_container_config(Path(args.config))
        except ConfigError as exc:
            print(f"Dependency scan failed: {exc}")
            return 1
        dependency_scan_result = scan_dependencies(
            container_config.scanner_timeout_seconds,
            container_config.container_runtime_requirements,
        )
        if dependency_scan_result.available and dependency_scan_result.status == "PASS":
            dependency_scan_result.status = (
                "PASS"
                if severity_gate(
                    dependency_scan_result.findings,
                    container_config.vulnerability_fail_severities,
                )
                else "FAIL"
            )
        write_tool_evidence(container_config, "pip-audit", dependency_scan_result)
        if args.json:
            print(
                json.dumps(dependency_scan_result.model_dump(mode="json"), indent=2, sort_keys=True)
            )
        else:
            print(
                f"Dependency scan status={dependency_scan_result.status} "
                f"tool={dependency_scan_result.tool}."
            )
            if dependency_scan_result.output:
                print(dependency_scan_result.output)
        return 0 if dependency_scan_result.status == "PASS" else 2

    if args.command == "generate-release-sbom":
        from medical_imaging_platform.release.evidence import write_tool_evidence
        from medical_imaging_platform.release.scanners import generate_context_sbom

        try:
            container_config = load_container_config(Path(args.config))
        except ConfigError as exc:
            print(f"SBOM generation failed: {exc}")
            return 1
        output_root = (
            Path(args.output_dir)
            if args.output_dir is not None
            else container_config.release_output_directory / "sbom-latest"
        )
        api_result = generate_context_sbom(
            f"{container_config.api_image_name}:{container_config.image_tag}",
            output_root / "api.sbom.cdx.json",
            container_config.scanner_timeout_seconds,
        )
        ui_result = generate_context_sbom(
            f"{container_config.reviewer_ui_image_name}:{container_config.image_tag}",
            output_root / "reviewer-ui.sbom.cdx.json",
            container_config.scanner_timeout_seconds,
        )
        write_tool_evidence(
            container_config,
            "syft-api",
            api_result.model_copy(update={"details": {**api_result.details, "target": "api"}}),
        )
        write_tool_evidence(
            container_config,
            "syft-reviewer-ui",
            ui_result.model_copy(
                update={"details": {**ui_result.details, "target": "reviewer-ui"}}
            ),
        )
        sbom_payload = [api_result.model_dump(mode="json"), ui_result.model_dump(mode="json")]
        if args.json:
            print(json.dumps(sbom_payload, indent=2, sort_keys=True))
        else:
            print(f"SBOM generation statuses={[api_result.status, ui_result.status]}.")
        sbom_statuses = {api_result.status, ui_result.status}
        return 0 if sbom_statuses == {"PASS"} else 2

    if args.command == "scan-release-images":
        from medical_imaging_platform.release.evidence import write_tool_evidence
        from medical_imaging_platform.release.scanners import scan_image, severity_gate

        try:
            container_config = load_container_config(Path(args.config))
        except ConfigError as exc:
            print(f"Image scan failed: {exc}")
            return 1
        refs = [
            f"{container_config.api_image_name}:{container_config.image_tag}",
            f"{container_config.reviewer_ui_image_name}:{container_config.image_tag}",
        ]
        image_scan_results = [
            scan_image(ref, container_config.scanner_timeout_seconds) for ref in refs
        ]
        for image_scan_result in image_scan_results:
            if image_scan_result.available and image_scan_result.status == "PASS":
                image_scan_result.status = (
                    "PASS"
                    if severity_gate(
                        image_scan_result.findings,
                        container_config.vulnerability_fail_severities,
                    )
                    else "FAIL"
                )
        for key, target, image_scan_result in zip(
            ("trivy-api", "trivy-reviewer-ui"),
            ("api", "reviewer-ui"),
            image_scan_results,
            strict=True,
        ):
            write_tool_evidence(
                container_config,
                key,
                image_scan_result.model_copy(
                    update={"details": {**image_scan_result.details, "target": target}}
                ),
            )
        if args.json:
            print(
                json.dumps(
                    [scan_result.model_dump(mode="json") for scan_result in image_scan_results],
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(
                f"Image scan statuses={[scan_result.status for scan_result in image_scan_results]}."
            )
        image_scan_statuses = {scan_result.status for scan_result in image_scan_results}
        return 0 if image_scan_statuses == {"PASS"} else 2

    if args.command == "register-model":
        from medical_imaging_platform.governance.models import ModelType
        from medical_imaging_platform.governance.registry import (
            REGISTRY_PATH,
            build_model_record,
            load_registry,
            register_model,
        )

        path = Path(args.registry_path) if args.registry_path is not None else REGISTRY_PATH
        try:
            model_types: tuple[tuple[ModelType, str, str], ...] = (
                ("segmentation", "m14-segmentation-synthetic-v1", "config/segmentation.yaml"),
                ("classification", "m14-classification-synthetic-v1", "config/classification.yaml"),
            )
            selected = [
                item
                for item in model_types
                if args.model_type == "all" or item[0] == args.model_type
            ]
            registry_manifest = load_registry(path)
            for model_type, version, config_reference in selected:
                if any(
                    record.model_type == model_type and record.version == version
                    for record in registry_manifest.records
                ):
                    continue
                registry_manifest = register_model(
                    build_model_record(
                        model_type=model_type,
                        version=version,
                        checkpoint_reference=f"synthetic://{model_type}/best_model.pt",
                        config_reference=config_reference,
                        training_data_reference=f"synthetic://m14/{model_type}/dataset",
                    ),
                    path,
                )
        except ValueError as exc:
            print(f"Model registration failed: {exc}")
            return 1
        if args.json:
            print(json.dumps(registry_manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(f"Registered/listed {len(registry_manifest.records)} model records at {path}.")
        return 0

    if args.command == "list-models":
        from medical_imaging_platform.governance.registry import REGISTRY_PATH, load_registry

        path = Path(args.registry_path) if args.registry_path is not None else REGISTRY_PATH
        registry_manifest = load_registry(path)
        if args.json:
            print(json.dumps(registry_manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(f"Registry contains {len(registry_manifest.records)} model records.")
            for record in registry_manifest.records:
                print(
                    f"{record.model_name} {record.version} "
                    f"type={record.model_type} state={record.lifecycle_state}"
                )
        return 0

    if args.command == "approve-model":
        from medical_imaging_platform.governance.models import ApprovalMetadata
        from medical_imaging_platform.governance.registry import REGISTRY_PATH, approve_model

        path = Path(args.registry_path) if args.registry_path is not None else REGISTRY_PATH
        try:
            registry_manifest = approve_model(
                args.model_name,
                args.version,
                ApprovalMetadata(
                    approved_by=args.approved_by,
                    approval_ticket=args.approval_ticket,
                    approval_timestamp=args.approval_timestamp,
                    rationale=args.rationale,
                ),
                path,
            )
        except ValueError as exc:
            print(f"Model approval failed: {exc}")
            return 1
        if args.json:
            print(json.dumps(registry_manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(f"Approved {args.model_name}:{args.version}.")
        return 0

    if args.command == "build-monitoring-baseline":
        from medical_imaging_platform.governance.monitoring import build_baseline
        from medical_imaging_platform.governance.registry import ensure_demo_registry

        ensure_demo_registry()
        baseline = build_baseline()
        if args.json:
            print(json.dumps(baseline.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(f"Built monitoring baseline {baseline.baseline_id}.")
        return 0

    if args.command == "run-monitoring":
        from medical_imaging_platform.governance.monitoring import run_monitoring

        run = run_monitoring(mode="normal")
        if args.json:
            print(json.dumps(run.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(f"Monitoring run {run.run_id} status={run.overall_status}.")
        return 0 if run.overall_status == "PASS" else 2

    if args.command == "simulate-monitoring-drift":
        from medical_imaging_platform.governance.monitoring import run_monitoring

        run = run_monitoring(mode="simulated_drift")
        if args.json:
            print(json.dumps(run.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            print(f"Simulated monitoring drift {run.run_id} status={run.overall_status}.")
        return 0

    if args.command == "build-audit-evidence":
        from medical_imaging_platform.governance.audit import build_audit_evidence

        events = build_audit_evidence()
        if args.json:
            print(json.dumps([event.model_dump(mode="json") for event in events], indent=2))
        else:
            print(f"Built audit evidence with {len(events)} events.")
        return 0

    if args.command == "validate-monitoring-evidence":
        from medical_imaging_platform.governance.monitoring import validate_monitoring_evidence

        validation_checks = validate_monitoring_evidence()
        status = "PASS" if all(check["status"] == "PASS" for check in validation_checks) else "FAIL"
        if args.json:
            print(
                json.dumps(
                    {"status": status, "checks": validation_checks},
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"Monitoring evidence validation status={status}.")
        return 0 if status == "PASS" else 2

    if args.command in {
        "validate-helm",
        "render-kubernetes",
        "validate-kubernetes-policy",
        "deploy-local-kubernetes",
        "kubernetes-smoke",
        "build-kubernetes-evidence",
        "validate-kubernetes-evidence",
        "clean-local-kubernetes",
    }:
        from medical_imaging_platform.kubernetes.assurance import (
            build_kubernetes_evidence,
            clean_local_kubernetes,
            deploy_local_kubernetes,
            kubernetes_smoke,
            render_kubernetes_manifests,
            validate_helm_chart,
            validate_kubernetes_evidence,
            validate_kubernetes_policy,
            validate_values_schema,
        )

        if args.command == "validate-helm":
            helm_result = validate_helm_chart()
            schema_result = validate_values_schema()
            payload = {"helm_lint": helm_result, "schema_validation": schema_result}
            if args.json:
                print(
                    json.dumps(
                        {key: value.model_dump(mode="json") for key, value in payload.items()},
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(f"Helm validation helm={helm_result.status} schema={schema_result.status}.")
            return 0 if schema_result.status == "PASS" and helm_result.status != "FAIL" else 2
        if args.command == "render-kubernetes":
            manifests = render_kubernetes_manifests()
            if args.json:
                print(json.dumps({"rendered_objects": len(manifests)}, indent=2))
            else:
                print(f"Rendered {len(manifests)} Kubernetes objects.")
            return 0
        if args.command == "validate-kubernetes-policy":
            kubernetes_policy_checks = validate_kubernetes_policy()
            status = (
                "PASS"
                if all(check.status == "PASS" for check in kubernetes_policy_checks)
                else "FAIL"
            )
            if args.json:
                print(
                    json.dumps(
                        {
                            "status": status,
                            "checks": [
                                check.model_dump(mode="json") for check in kubernetes_policy_checks
                            ],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(f"Kubernetes policy validation status={status}.")
            return 0 if status == "PASS" else 2
        if args.command == "deploy-local-kubernetes":
            kubernetes_runtime_result = deploy_local_kubernetes()
        elif args.command == "kubernetes-smoke":
            kubernetes_runtime_result = kubernetes_smoke()
        elif args.command == "clean-local-kubernetes":
            kubernetes_runtime_result = clean_local_kubernetes()
        elif args.command == "build-kubernetes-evidence":
            kubernetes_manifest = build_kubernetes_evidence()
            if args.json:
                print(
                    json.dumps(
                        kubernetes_manifest.model_dump(mode="json"),
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(f"Built Kubernetes evidence status={kubernetes_manifest.overall_status}.")
            return 0 if kubernetes_manifest.overall_status != "FAIL" else 2
        else:
            kubernetes_evidence_checks = validate_kubernetes_evidence()
            status = (
                "PASS"
                if all(check.status == "PASS" for check in kubernetes_evidence_checks)
                else "FAIL"
            )
            if args.json:
                print(
                    json.dumps(
                        {
                            "status": status,
                            "checks": [
                                check.model_dump(mode="json")
                                for check in kubernetes_evidence_checks
                            ],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(f"Kubernetes evidence validation status={status}.")
            return 0 if status == "PASS" else 2
        if args.json:
            print(
                json.dumps(
                    kubernetes_runtime_result.model_dump(mode="json"),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(
                "Kubernetes runtime result "
                f"status={kubernetes_runtime_result.status} "
                f"executed={kubernetes_runtime_result.executed}."
            )
        return 0 if kubernetes_runtime_result.status in {"PASS", "UNAVAILABLE", "INCOMPLETE"} else 2

    if args.command in {
        "terraform-fmt-check",
        "terraform-init",
        "terraform-validate",
        "validate-aws-policy",
        "scan-terraform",
        "build-aws-evidence",
        "validate-aws-evidence",
        "clean-terraform",
        "aws-plan",
    }:
        from medical_imaging_platform.aws.assurance import (
            aws_plan,
            build_aws_evidence,
            clean_terraform,
            scan_terraform,
            terraform_fmt_check,
            terraform_init,
            terraform_validate,
            validate_aws_evidence,
            validate_aws_policy,
        )

        if args.command in {
            "terraform-fmt-check",
            "terraform-init",
            "terraform-validate",
            "clean-terraform",
            "aws-plan",
        }:
            aws_check = {
                "terraform-fmt-check": terraform_fmt_check,
                "terraform-init": terraform_init,
                "terraform-validate": terraform_validate,
                "clean-terraform": clean_terraform,
                "aws-plan": aws_plan,
            }[args.command]()
            if args.json:
                print(json.dumps(aws_check.model_dump(mode="json"), indent=2, sort_keys=True))
            else:
                print(f"{aws_check.check_id} status={aws_check.status}.")
            return 0 if aws_check.status in {"PASS", "UNAVAILABLE", "INCOMPLETE"} else 2
        if args.command == "validate-aws-policy":
            aws_policy_checks = validate_aws_policy()
            status = (
                "PASS" if all(check.status == "PASS" for check in aws_policy_checks) else "FAIL"
            )
            if args.json:
                print(
                    json.dumps(
                        {
                            "status": status,
                            "checks": [
                                check.model_dump(mode="json") for check in aws_policy_checks
                            ],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(f"AWS policy validation status={status}.")
            return 0 if status == "PASS" else 2
        if args.command == "scan-terraform":
            aws_scan_checks = scan_terraform()
            status = (
                "PASS" if all(check.status == "PASS" for check in aws_scan_checks) else "INCOMPLETE"
            )
            if any(check.status in {"FAIL", "ERROR"} for check in aws_scan_checks):
                status = "FAIL"
            if args.json:
                print(
                    json.dumps(
                        {
                            "status": status,
                            "checks": [check.model_dump(mode="json") for check in aws_scan_checks],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(f"Terraform scanner status={status}.")
            return 0 if status in {"PASS", "INCOMPLETE"} else 2
        if args.command == "build-aws-evidence":
            aws_manifest = build_aws_evidence()
            if args.json:
                print(json.dumps(aws_manifest.model_dump(mode="json"), indent=2, sort_keys=True))
            else:
                print(f"Built AWS evidence status={aws_manifest.overall_status}.")
            return 0 if aws_manifest.overall_status != "FAIL" else 2
        aws_evidence_checks = validate_aws_evidence()
        status = "PASS" if all(check.status == "PASS" for check in aws_evidence_checks) else "FAIL"
        if args.json:
            print(
                json.dumps(
                    {
                        "status": status,
                        "checks": [check.model_dump(mode="json") for check in aws_evidence_checks],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"AWS evidence validation status={status}.")
        return 0 if status == "PASS" else 2

    if args.command in {
        "validate-observability",
        "build-observability-evidence",
        "evaluate-slos",
        "simulate-incidents",
        "build-incident-evidence",
        "validate-runbooks",
        "simulate-rollback",
        "validate-recovery",
        "build-operations-evidence",
        "validate-operations-evidence",
    }:
        from medical_imaging_platform.operations.assurance import (
            build_incident_evidence,
            build_observability_evidence,
            build_operations_evidence,
            evaluate_slos,
            simulate_incidents,
            simulate_rollback,
            validate_observability,
            validate_operations_evidence,
            validate_recovery,
            validate_runbooks,
        )

        if args.command == "build-observability-evidence":
            payload = build_observability_evidence()
            print(
                json.dumps(payload, indent=2, sort_keys=True)
                if args.json
                else "Built observability evidence."
            )
            return 0
        if args.command == "evaluate-slos":
            operations_slo_checks = evaluate_slos()
            operations_slo_status = (
                "PASS"
                if all(check.status == "PASS" for check in operations_slo_checks)
                else "ALERT"
            )
            print(
                json.dumps(
                    [check.model_dump(mode="json") for check in operations_slo_checks], indent=2
                )
                if args.json
                else f"SLO evaluation status={operations_slo_status}."
            )
            return 0 if operations_slo_status == "PASS" else 2
        if args.command == "simulate-incidents":
            incidents = simulate_incidents()
            print(
                json.dumps([incident.model_dump(mode="json") for incident in incidents], indent=2)
                if args.json
                else f"Simulated {len(incidents)} incidents."
            )
            return 0
        if args.command == "build-incident-evidence":
            records = build_incident_evidence()
            print(
                json.dumps(records, indent=2, sort_keys=True)
                if args.json
                else f"Built {len(records)} incident lifecycle records."
            )
            return 0
        if args.command == "simulate-rollback":
            rollback = simulate_rollback()
            print(
                json.dumps(rollback, indent=2, sort_keys=True)
                if args.json
                else f"Rollback simulation status={rollback['status']}."
            )
            return 0
        if args.command in {"validate-observability", "validate-runbooks", "validate-recovery"}:
            operations_validation_checks = {
                "validate-observability": validate_observability,
                "validate-runbooks": validate_runbooks,
                "validate-recovery": validate_recovery,
            }[args.command]()
            operations_validation_status = (
                "PASS"
                if all(check.status == "PASS" for check in operations_validation_checks)
                else "FAIL"
            )
            print(
                json.dumps(
                    [check.model_dump(mode="json") for check in operations_validation_checks],
                    indent=2,
                )
                if args.json
                else f"{args.command} status={operations_validation_status}."
            )
            return 0 if operations_validation_status == "PASS" else 2
        if args.command == "build-operations-evidence":
            operations_manifest = build_operations_evidence()
            print(
                json.dumps(operations_manifest.model_dump(mode="json"), indent=2, sort_keys=True)
                if args.json
                else f"Built operations evidence status={operations_manifest.overall_status}."
            )
            return 0 if operations_manifest.overall_status == "PASS" else 2
        operations_evidence_checks = validate_operations_evidence()
        operations_evidence_status = (
            "PASS"
            if all(check.status == "PASS" for check in operations_evidence_checks)
            else "FAIL"
        )
        print(
            json.dumps(
                [check.model_dump(mode="json") for check in operations_evidence_checks], indent=2
            )
            if args.json
            else f"Operations evidence validation status={operations_evidence_status}."
        )
        return 0 if operations_evidence_status == "PASS" else 2

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


def _resolve_release_dir(path: Path | None) -> Path:
    if path is not None:
        return path
    release_root = Path("reports/generated/releases")
    candidates = sorted(
        item
        for item in release_root.glob("*")
        if item.is_dir() and (item / "release_manifest.json").is_file()
    )
    if not candidates:
        raise ValueError("No generated release evidence directory found.")
    return candidates[-1]
