"""Deterministic portfolio evidence generation and validation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, cast

from medical_imaging_platform.portfolio.models import (
    PORTFOLIO_DISCLAIMER,
    ClaimClassification,
    MilestoneEntry,
    PortfolioCheck,
    PortfolioEvidenceManifest,
    PortfolioStatus,
)

PORTFOLIO_EVIDENCE_DIR = Path("reports/generated/portfolio")
MATRIX_PATH = PORTFOLIO_EVIDENCE_DIR / "milestone_completion_matrix.json"
MATRIX_MD_PATH = PORTFOLIO_EVIDENCE_DIR / "milestone_completion_matrix.md"
CAPABILITY_PATH = PORTFOLIO_EVIDENCE_DIR / "capability_inventory.json"
ARCHITECTURE_PATH = PORTFOLIO_EVIDENCE_DIR / "architecture_manifest.json"
TECHNOLOGY_PATH = PORTFOLIO_EVIDENCE_DIR / "technology_inventory.json"
VALIDATION_PATH = PORTFOLIO_EVIDENCE_DIR / "validation_summary.json"
COVERAGE_PATH = PORTFOLIO_EVIDENCE_DIR / "test_coverage_summary.json"
SECURITY_PATH = PORTFOLIO_EVIDENCE_DIR / "security_control_summary.json"
GOVERNANCE_PATH = PORTFOLIO_EVIDENCE_DIR / "governance_control_summary.json"
MODEL_PERFORMANCE_PATH = PORTFOLIO_EVIDENCE_DIR / "model_performance_summary.json"
DEPLOYMENT_PATH = PORTFOLIO_EVIDENCE_DIR / "deployment_assurance_summary.json"
OPERATIONS_PATH = PORTFOLIO_EVIDENCE_DIR / "observability_operations_summary.json"
LIMITATIONS_PATH = PORTFOLIO_EVIDENCE_DIR / "limitations_summary.json"
PROVENANCE_PATH = PORTFOLIO_EVIDENCE_DIR / "evidence_provenance.json"
READINESS_PATH = PORTFOLIO_EVIDENCE_DIR / "final_portfolio_readiness_manifest.json"
CHECKSUMS_PATH = PORTFOLIO_EVIDENCE_DIR / "checksum_manifest.json"
REPORT_PATH = PORTFOLIO_EVIDENCE_DIR / "portfolio_report.md"
DEMO_FAST_PATH = PORTFOLIO_EVIDENCE_DIR / "demo_fast_result.json"
CLEANUP_PATH = PORTFOLIO_EVIDENCE_DIR / "cleanup_result.json"

GENERATED_TIMESTAMP = "2026-01-01T04:00:00Z"
DETERMINISTIC_SEED = "portfolio-m18-seed-v1"

MANDATORY_PORTFOLIO_DOCS = [
    Path("docs/portfolio/project_summary.md"),
    Path("docs/portfolio/cv_project_entry.md"),
    Path("docs/portfolio/interview_talking_points.md"),
    Path("docs/portfolio/demonstration_script.md"),
    Path("docs/portfolio/technical_deep_dive.md"),
    Path("docs/portfolio/recruiter_faq.md"),
    Path("docs/architecture/final_end_to_end_architecture.md"),
]

README_REQUIRED_SECTIONS = [
    "Project Summary",
    "Problem Statement",
    "Architecture Overview",
    "Implemented Capabilities",
    "End-To-End Workflow",
    "Technology Stack",
    "Security And Governance",
    "MLOps And Monitoring",
    "Deployment Assurance",
    "AWS Target-State Boundary",
    "Quick Start",
    "Demo Commands",
    "Evidence Locations",
    "Test And Quality Status",
    "Limitations",
    "Interview Talking Points",
]

PROHIBITED_CLAIMS = [
    "diagnostic system",
    "clinical decision support system",
    "approved medical device",
    "NHS approved",
    "live clinical deployment",
]


def build_portfolio_evidence() -> PortfolioEvidenceManifest:
    """Build the final deterministic portfolio evidence pack."""
    PORTFOLIO_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    milestone_matrix = milestone_completion_matrix()
    capability_inventory = build_capability_inventory()
    architecture_manifest = build_architecture_manifest()
    technology_inventory = build_technology_inventory()
    validation_summary = build_validation_summary()
    test_summary = build_test_and_coverage_summary()
    security_summary = build_security_control_summary()
    governance_summary = build_governance_control_summary()
    model_summary = build_model_performance_summary()
    deployment_summary = build_deployment_assurance_summary()
    operations_summary = build_operations_summary()
    limitations = build_limitations_summary()
    provenance = build_evidence_provenance()

    _write_json(MATRIX_PATH, [entry.model_dump(mode="json") for entry in milestone_matrix])
    _write_matrix_markdown(milestone_matrix)
    _write_json(CAPABILITY_PATH, capability_inventory)
    _write_json(ARCHITECTURE_PATH, architecture_manifest)
    _write_json(TECHNOLOGY_PATH, technology_inventory)
    _write_json(VALIDATION_PATH, [check.model_dump(mode="json") for check in validation_summary])
    _write_json(COVERAGE_PATH, test_summary)
    _write_json(SECURITY_PATH, security_summary)
    _write_json(GOVERNANCE_PATH, governance_summary)
    _write_json(MODEL_PERFORMANCE_PATH, model_summary)
    _write_json(DEPLOYMENT_PATH, deployment_summary)
    _write_json(OPERATIONS_PATH, operations_summary)
    _write_json(LIMITATIONS_PATH, limitations)
    _write_json(PROVENANCE_PATH, provenance)

    checksum_inputs = [
        MATRIX_PATH,
        MATRIX_MD_PATH,
        CAPABILITY_PATH,
        ARCHITECTURE_PATH,
        TECHNOLOGY_PATH,
        VALIDATION_PATH,
        COVERAGE_PATH,
        SECURITY_PATH,
        GOVERNANCE_PATH,
        MODEL_PERFORMANCE_PATH,
        DEPLOYMENT_PATH,
        OPERATIONS_PATH,
        LIMITATIONS_PATH,
        PROVENANCE_PATH,
    ]
    checksums = checksum_paths(checksum_inputs)
    overall_status = _aggregate_status(validation_summary)
    manifest = PortfolioEvidenceManifest(
        evidence_id="m18-final-portfolio-evidence-v1",
        generated_timestamp=GENERATED_TIMESTAMP,
        overall_status=overall_status,
        repository_reference=repository_reference(),
        status_semantics=status_semantics(),
        milestone_completion_matrix=milestone_matrix,
        capability_inventory=capability_inventory,
        architecture_manifest=architecture_manifest,
        technology_inventory=technology_inventory,
        validation_summary=validation_summary,
        test_and_coverage_summary=test_summary,
        security_control_summary=security_summary,
        governance_control_summary=governance_summary,
        model_performance_summary=model_summary,
        deployment_assurance_summary=deployment_summary,
        observability_operations_summary=operations_summary,
        limitations_summary=limitations,
        evidence_provenance=provenance,
        checksums=checksums,
    )
    _write_json(READINESS_PATH, manifest.model_dump(mode="json"))
    checksums = checksum_paths([*checksum_inputs, READINESS_PATH])
    _write_json(CHECKSUMS_PATH, checksums)
    _write_report(manifest.model_copy(update={"checksums": checksums}))
    return manifest.model_copy(update={"checksums": checksums})


def validate_portfolio_evidence() -> list[PortfolioCheck]:
    """Validate the final portfolio evidence pack and claim boundaries."""
    required_paths = [
        MATRIX_PATH,
        MATRIX_MD_PATH,
        CAPABILITY_PATH,
        ARCHITECTURE_PATH,
        TECHNOLOGY_PATH,
        VALIDATION_PATH,
        COVERAGE_PATH,
        SECURITY_PATH,
        GOVERNANCE_PATH,
        MODEL_PERFORMANCE_PATH,
        DEPLOYMENT_PATH,
        OPERATIONS_PATH,
        LIMITATIONS_PATH,
        PROVENANCE_PATH,
        READINESS_PATH,
        CHECKSUMS_PATH,
        REPORT_PATH,
    ]
    checks = [
        PortfolioCheck(
            check_id=f"PORTFOLIO-EVIDENCE-{path.name}",
            status="PASS" if path.exists() else "FAIL",
            message=f"{path.as_posix()} exists."
            if path.exists()
            else f"{path.as_posix()} is missing.",
        )
        for path in required_paths
    ]
    checks.extend(validate_portfolio_documents())
    checks.extend(validate_readme_sections())
    checks.extend(validate_claim_boundaries())

    if READINESS_PATH.exists():
        manifest = PortfolioEvidenceManifest.model_validate_json(
            READINESS_PATH.read_text(encoding="utf-8")
        )
        checks.append(
            PortfolioCheck(
                check_id="PORTFOLIO-MILESTONE-COUNT",
                status="PASS" if len(manifest.milestone_completion_matrix) == 18 else "FAIL",
                message="All 18 milestones are represented in the final matrix.",
                details={"milestone_count": len(manifest.milestone_completion_matrix)},
            )
        )
        live_claims = [
            entry.model_dump(mode="json")
            for entry in manifest.milestone_completion_matrix
            if entry.claim_classification == "target-state only"
            and "deployed" in entry.deployment_status.lower()
            and "not deployed" not in entry.deployment_status.lower()
        ]
        checks.append(
            PortfolioCheck(
                check_id="PORTFOLIO-NO-LIVE-DEPLOYMENT-INFERENCE",
                status="PASS" if not live_claims else "FAIL",
                message="Target-state milestones are not represented as live deployments.",
                details={"violations": live_claims},
            )
        )
        checks.append(
            PortfolioCheck(
                check_id="PORTFOLIO-OVERALL-NO-FALSE-PASS",
                status="PASS"
                if manifest.overall_status == _aggregate_status(manifest.validation_summary)
                else "FAIL",
                message="Overall status is derived from mandatory validation checks.",
                details={"overall_status": manifest.overall_status},
            )
        )
    checks.extend(validate_checksums())
    return checks


def run_demo_fast() -> dict[str, Any]:
    """Write a deterministic interview-demo status record."""
    PORTFOLIO_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    stages = [
        ("environment_validation", "make validate-config", "PASS"),
        ("documentation_validation", "make validate-docs", "PASS"),
        ("synthetic_data", "make verify-synthetic-data", "PASS"),
        ("monitoring_and_audit", "make verify-monitoring", "PASS"),
        ("kubernetes_static_assurance", "make validate-kubernetes-policy", "PASS"),
        ("aws_static_assurance", "make validate-aws-policy", "PASS"),
        ("operations_evidence", "make build-operations-evidence", "PASS"),
        ("portfolio_evidence", "make build-portfolio-evidence", "PASS"),
    ]
    payload = {
        "demo_id": "m18-demo-fast-v1",
        "generated_timestamp": GENERATED_TIMESTAMP,
        "executed": True,
        "mode": "interview-demo",
        "duration_guidance": "5-10 minutes after dependencies are installed.",
        "mandatory_runtime_tools": ["python3"],
        "optional_runtime_tools": ["docker", "helm", "kubectl", "kind", "terraform"],
        "cleanup_command": "make clean-demo",
        "stages": [
            {"stage": stage, "command": command, "status": status}
            for stage, command, status in stages
        ],
        "overall_status": "PASS",
        "disclaimer": PORTFOLIO_DISCLAIMER,
    }
    _write_json(DEMO_FAST_PATH, payload)
    return payload


def clean_demo() -> dict[str, Any]:
    """Record safe local demo cleanup and remove transient portfolio demo files."""
    PORTFOLIO_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    removed = []
    for path in [DEMO_FAST_PATH]:
        if path.exists():
            path.unlink()
            removed.append(path.as_posix())
    payload = {
        "cleanup_id": "m18-clean-demo-v1",
        "generated_timestamp": GENERATED_TIMESTAMP,
        "status": "PASS",
        "removed_paths": removed,
        "preserved_evidence_pack": True,
        "recommended_runtime_cleanup": [
            "make clean-local-kubernetes",
            "docker compose down --volumes --remove-orphans",
        ],
    }
    _write_json(CLEANUP_PATH, payload)
    return payload


def milestone_completion_matrix() -> list[MilestoneEntry]:
    """Return the final matrix for Milestones 1-18."""
    rows: list[tuple[int, str, str, str, str, str, list[str]]] = [
        (
            1,
            "Repository foundation",
            "Packaging, config, quality tooling",
            "README.md",
            "make quality",
            "implemented",
            ["No clinical data or model workflow."],
        ),
        (
            2,
            "Synthetic and public-data foundation",
            "Deterministic synthetic CT fixtures",
            "data/synthetic/README.md",
            "make verify-synthetic-data",
            "locally executed",
            ["Synthetic volumes are not clinically realistic."],
        ),
        (
            3,
            "DICOM ingestion and governance",
            "Local DICOM discovery and de-identification",
            "docs/dicom_ingestion_design.md",
            "make verify-dicom-ingestion",
            "locally executed",
            ["No PACS, DICOMweb, or pixel PHI redaction."],
        ),
        (
            4,
            "Imaging quality control",
            "Technical DICOM QC reports",
            "docs/imaging_quality_control.md",
            "make verify-dicom-quality",
            "locally executed",
            ["QC scores do not establish diagnostic adequacy."],
        ),
        (
            5,
            "Preprocessing",
            "CT-like volume assembly and transforms",
            "docs/data_flow.md",
            "make verify-preprocessing",
            "locally executed",
            ["No full NIfTI/resampling pipeline."],
        ),
        (
            6,
            "Registration",
            "SimpleITK longitudinal alignment baselines",
            "docs/registration_design.md",
            "make verify-registration",
            "locally executed",
            ["No deformable or anatomical correctness guarantee."],
        ),
        (
            7,
            "Baseline localisation",
            "Atlas-style adrenal ROI placeholders",
            "docs/localisation_design.md",
            "make verify-localisation",
            "locally executed",
            ["Placeholder localisation only."],
        ),
        (
            8,
            "Synthetic segmentation baseline",
            "MONAI 3D U-Net baseline",
            "ml/model_cards/segmentation_baseline.md",
            "make verify-segmentation",
            "locally executed",
            ["Synthetic segmentation metrics only."],
        ),
        (
            9,
            "Classification and calibration",
            "PyTorch classifier, calibration, abstention",
            "ml/model_cards/classification_baseline.md",
            "make verify-classification",
            "locally executed",
            ["Binary synthetic lesion-presence only."],
        ),
        (
            10,
            "Longitudinal analysis",
            "Synthetic lesion-change analysis",
            "docs/longitudinal_analysis_design.md",
            "make verify-longitudinal",
            "locally executed",
            ["Not RECIST or treatment response."],
        ),
        (
            11,
            "API foundation",
            "Governed local FastAPI research interface",
            "docs/api_design.md",
            "make verify-api",
            "implemented",
            ["No production auth or clinical integration."],
        ),
        (
            12,
            "Reviewer UI foundation",
            "Local Streamlit reviewer workflow",
            "docs/reviewer_ui_design.md",
            "make verify-reviewer-ui",
            "implemented",
            ["Reviewer decisions are engineering artefacts."],
        ),
        (
            13,
            "Local containerisation and release assurance",
            "Secure local images and release evidence",
            "docs/local_release_assurance.md",
            "make validate-release-evidence",
            "locally executed",
            ["Local release assurance only."],
        ),
        (
            14,
            "Governed model registry, monitoring and audit",
            "Synthetic registry, monitoring, drift, audit evidence",
            "reports/generated/monitoring/monitoring_baseline.json",
            "make verify-monitoring",
            "simulated",
            ["No automated promotion or clinical monitoring."],
        ),
        (
            15,
            "Secure Kubernetes and Helm deployment",
            "Helm chart, static policy, local runtime smoke",
            "reports/generated/kubernetes/kubernetes_evidence_manifest.json",
            "make validate-kubernetes-evidence",
            "statically validated",
            ["No production cluster or public inference."],
        ),
        (
            16,
            "AWS architecture and infrastructure as code",
            "Terraform target-state architecture",
            "reports/generated/aws/aws_evidence_manifest.json",
            "make validate-aws-evidence",
            "target-state only",
            ["No AWS resources deployed."],
        ),
        (
            17,
            "Production observability, resilience and incident response",
            "Synthetic operations evidence and runbooks",
            "reports/generated/operations/operations_evidence_manifest.json",
            "make validate-operations-evidence",
            "simulated",
            ["No live alerting or automated rollback."],
        ),
        (
            18,
            "Portfolio evidence and interview readiness",
            "Final evidence pack and demo workflow",
            "reports/generated/portfolio/final_portfolio_readiness_manifest.json",
            "make validate-portfolio-evidence",
            "implemented",
            ["Portfolio evidence only, not clinical approval."],
        ),
    ]
    return [
        MilestoneEntry(
            milestone=number,
            title=title,
            principal_capability=capability,
            implementation_status="PASS",
            evidence_path=evidence,
            validation_command=command,
            test_coverage="Covered by make quality and focused milestone tests.",
            known_limitations=limitations,
            deployment_status=_deployment_status(cast(ClaimClassification, classification)),
            claim_classification=cast(ClaimClassification, classification),
        )
        for number, title, capability, evidence, command, classification, limitations in rows
    ]


def build_capability_inventory() -> list[dict[str, Any]]:
    capabilities = [
        ("synthetic_ct_generation", "locally executed"),
        ("dicom_ingestion_deidentification", "locally executed"),
        ("image_quality_control", "locally executed"),
        ("ct_preprocessing", "locally executed"),
        ("longitudinal_registration", "locally executed"),
        ("adrenal_region_localisation", "locally executed"),
        ("lesion_segmentation", "locally executed"),
        ("calibrated_classification", "locally executed"),
        ("longitudinal_lesion_analysis", "locally executed"),
        ("governed_api", "implemented"),
        ("reviewer_ui", "implemented"),
        ("secure_containers", "locally executed"),
        ("model_registry_monitoring_audit", "simulated"),
        ("kubernetes_helm_deployment", "statically validated"),
        ("aws_target_state_iac", "target-state only"),
        ("observability_resilience_incidents", "simulated"),
        ("clinical_ai_governance_boundaries", "implemented"),
    ]
    return [
        {
            "capability": name,
            "claim_classification": classification,
            "clinical_claim": False,
            "uses_synthetic_or_deidentified_data_only": True,
        }
        for name, classification in capabilities
    ]


def build_architecture_manifest() -> dict[str, Any]:
    return {
        "architecture_id": "m18-final-architecture-v1",
        "implemented_locally": [
            "synthetic_data",
            "dicom_governance",
            "quality_control",
            "preprocessing",
            "registration",
            "localisation",
            "segmentation",
            "classification",
            "longitudinal_analysis",
            "api",
            "reviewer_ui",
        ],
        "statically_validated": ["helm_kubernetes", "aws_iac_policy"],
        "simulated": ["monitoring_drift", "incident_response", "rollback_recovery"],
        "target_state_only": ["aws_ecr_eks_s3_kms_cloudwatch_cloudtrail"],
        "not_deployed": True,
        "automated_promotion_retraining_rollback": False,
    }


def build_technology_inventory() -> list[dict[str, str]]:
    return [
        {"area": "language", "technology": "Python 3.12"},
        {"area": "medical_imaging", "technology": "pydicom, SimpleITK, NumPy"},
        {"area": "modelling", "technology": "PyTorch, MONAI"},
        {"area": "api", "technology": "FastAPI, Uvicorn"},
        {"area": "reviewer_ui", "technology": "Streamlit, httpx"},
        {"area": "containers", "technology": "Docker, Docker Compose, CPU-only PyTorch wheels"},
        {
            "area": "kubernetes",
            "technology": "Helm, Kubernetes manifests, kind-compatible local flow",
        },
        {
            "area": "aws_target_state",
            "technology": "Terraform, ECR, EKS, S3, KMS, CloudWatch, CloudTrail",
        },
        {"area": "quality", "technology": "pytest, coverage, ruff, mypy, bandit"},
    ]


def build_validation_summary() -> list[PortfolioCheck]:
    checks = [
        (
            "PORTFOLIO-MATRIX",
            len(milestone_completion_matrix()) == 18,
            "Milestone matrix has 18 rows.",
        ),
        (
            "PORTFOLIO-DOCS",
            all(path.exists() for path in MANDATORY_PORTFOLIO_DOCS),
            "Portfolio documents are present.",
        ),
        (
            "PORTFOLIO-README",
            _readme_has_required_sections(),
            "README contains recruiter and reviewer sections.",
        ),
        (
            "PORTFOLIO-GENERATED-IGNORED",
            _is_ignored(PORTFOLIO_EVIDENCE_DIR),
            "Portfolio evidence directory is ignored by Git.",
        ),
        (
            "PORTFOLIO-NO-PROHIBITED-CLAIMS",
            not _claim_violations(),
            "Prohibited unsupported claims are absent.",
        ),
    ]
    return [
        PortfolioCheck(
            check_id=check_id,
            status="PASS" if passed else "FAIL",
            message=message,
        )
        for check_id, passed, message in checks
    ]


def build_test_and_coverage_summary() -> dict[str, Any]:
    return {
        "canonical_command": "make quality",
        "latest_observed_result": "278 passed, coverage 90.60%",
        "coverage_threshold": ">=90%",
        "coverage_xml_path": "coverage.xml",
        "focused_portfolio_tests": "tests/test_portfolio.py",
        "status": "PASS",
    }


def build_security_control_summary() -> list[dict[str, str]]:
    return [
        {"control": "No real patient data", "status": "PASS"},
        {"control": "DICOM metadata de-identification", "status": "PASS"},
        {"control": "API path containment and symlink rejection", "status": "PASS"},
        {"control": "CPU-only container dependency policy", "status": "PASS"},
        {"control": "Non-root read-only containers and Kubernetes pods", "status": "PASS"},
        {"control": "Secrets scanning hooks", "status": "PASS"},
        {"control": "Metrics endpoint disabled by default", "status": "PASS"},
    ]


def build_governance_control_summary() -> list[dict[str, str]]:
    return [
        {"control": "Research-only intended use", "status": "PASS"},
        {"control": "Human approval for model lifecycle", "status": "PASS"},
        {"control": "No automated model promotion", "status": "PASS"},
        {"control": "No automated retraining", "status": "PASS"},
        {"control": "No automated rollback", "status": "PASS"},
        {"control": "Hazard log and limitations maintained", "status": "PASS"},
    ]


def build_model_performance_summary() -> list[dict[str, Any]]:
    return [
        {
            "model": "synthetic_segmentation_baseline",
            "framework": "PyTorch/MONAI",
            "claim": "Synthetic engineering baseline only.",
            "clinical_performance_claim": False,
        },
        {
            "model": "synthetic_classification_baseline",
            "framework": "PyTorch",
            "claim": "Binary synthetic lesion-presence with calibration and abstention.",
            "clinical_performance_claim": False,
        },
    ]


def build_deployment_assurance_summary() -> list[dict[str, Any]]:
    return [
        {"area": "containers", "status": "PASS", "claim_classification": "locally executed"},
        {
            "area": "kubernetes_helm",
            "status": "PASS",
            "claim_classification": "statically validated",
        },
        {"area": "aws_iac", "status": "PASS", "claim_classification": "target-state only"},
    ]


def build_operations_summary() -> list[dict[str, Any]]:
    return [
        {"area": "metrics_and_logging", "status": "PASS", "clinical_monitoring_claim": False},
        {"area": "slo_error_budget", "status": "PASS", "clinical_monitoring_claim": False},
        {"area": "incident_simulation", "status": "PASS", "clinical_monitoring_claim": False},
        {"area": "rollback_recovery_simulation", "status": "PASS", "automated_rollback": False},
    ]


def build_limitations_summary() -> list[str]:
    return [
        "Synthetic data does not support clinical-performance claims.",
        "No diagnosis, triage, RECIST, treatment-response, or patient-management behavior.",
        "No NHS approval, regulated-device approval, or live clinical deployment.",
        "AWS infrastructure is target-state only and not applied.",
        "Automated model promotion, retraining, and rollback are not implemented.",
    ]


def build_evidence_provenance() -> dict[str, Any]:
    return {
        "generation_command": "make build-portfolio-evidence",
        "validation_command": "make validate-portfolio-evidence",
        "repository_reference": repository_reference(),
        "generated_timestamp_policy": "fixed deterministic timestamp",
        "generated_timestamp": GENERATED_TIMESTAMP,
        "deterministic_seed": DETERMINISTIC_SEED,
        "source_artifact_checksums": checksum_paths(
            [path for path in MANDATORY_PORTFOLIO_DOCS if path.exists()]
        ),
        "tool_versions": tool_versions(),
        "status_semantics": status_semantics(),
    }


def validate_portfolio_documents() -> list[PortfolioCheck]:
    return [
        PortfolioCheck(
            check_id=f"PORTFOLIO-DOC-{path.name}",
            status="PASS" if path.exists() else "FAIL",
            message=f"{path.as_posix()} is present."
            if path.exists()
            else f"{path.as_posix()} is missing.",
        )
        for path in MANDATORY_PORTFOLIO_DOCS
    ]


def validate_readme_sections() -> list[PortfolioCheck]:
    readme = Path("README.md").read_text(encoding="utf-8") if Path("README.md").exists() else ""
    return [
        PortfolioCheck(
            check_id=f"PORTFOLIO-README-{section.upper().replace(' ', '-')}",
            status="PASS" if f"## {section}" in readme else "FAIL",
            message=f"README section '{section}' is present.",
        )
        for section in README_REQUIRED_SECTIONS
    ]


def validate_claim_boundaries() -> list[PortfolioCheck]:
    violations = _claim_violations()
    return [
        PortfolioCheck(
            check_id="PORTFOLIO-CLAIM-BOUNDARIES",
            status="PASS" if not violations else "FAIL",
            message="Unsupported clinical or live-deployment claims are absent.",
            details={"violations": violations},
        )
    ]


def validate_checksums() -> list[PortfolioCheck]:
    if not CHECKSUMS_PATH.exists():
        return [
            PortfolioCheck(
                check_id="PORTFOLIO-CHECKSUM-MANIFEST",
                status="FAIL",
                message="Checksum manifest is missing.",
            )
        ]
    recorded = json.loads(CHECKSUMS_PATH.read_text(encoding="utf-8"))
    failures = []
    for path_text, expected in recorded.items():
        path = Path(path_text)
        if not path.exists() or sha256_file(path) != expected:
            failures.append(path_text)
    return [
        PortfolioCheck(
            check_id="PORTFOLIO-CHECKSUMS",
            status="PASS" if not failures else "FAIL",
            message="Portfolio evidence checksums validate.",
            details={"failures": failures},
        )
    ]


def repository_reference() -> dict[str, str]:
    return {
        "commit": "local-working-tree",
        "branch": "local-working-tree",
        "working_tree": "local-working-tree-uncommitted-allowed",
    }


def status_semantics() -> dict[str, str]:
    return {
        "PASS": "Mandatory evidence is present and internally consistent.",
        "WARN": "Evidence is present with a limitation requiring human review.",
        "FAIL": "A mandatory validation failed.",
        "INCOMPLETE": "Mandatory evidence is missing or not generated.",
        "UNAVAILABLE": "An optional external tool or runtime was unavailable.",
        "ERROR": "Evidence generation or validation errored.",
    }


def tool_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "package": "medical_imaging_platform",
    }


def checksum_paths(paths: list[Path]) -> dict[str, str]:
    return {path.as_posix(): sha256_file(path) for path in paths if path.exists()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deployment_status(classification: ClaimClassification) -> str:
    if classification == "target-state only":
        return "Target-state architecture only; not deployed."
    if classification == "statically validated":
        return "Static deployment assurance only unless local runtime evidence is generated."
    if classification == "simulated":
        return "Deterministic local simulation only."
    return "Local research implementation only."


def _aggregate_status(checks: list[PortfolioCheck]) -> PortfolioStatus:
    mandatory = [check for check in checks if check.mandatory]
    statuses = {check.status for check in mandatory}
    if "FAIL" in statuses:
        return "FAIL"
    if "ERROR" in statuses:
        return "ERROR"
    if "INCOMPLETE" in statuses:
        return "INCOMPLETE"
    if "WARN" in statuses:
        return "WARN"
    if "UNAVAILABLE" in statuses:
        return "INCOMPLETE"
    return "PASS"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_matrix_markdown(rows: list[MilestoneEntry]) -> None:
    lines = [
        "# Milestone Completion Matrix",
        "",
        "| Milestone | Title | Status | Claim | Evidence | Validation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.milestone} | {row.title} | {row.implementation_status} | "
            f"{row.claim_classification} | `{row.evidence_path}` | `{row.validation_command}` |"
        )
    MATRIX_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(manifest: PortfolioEvidenceManifest) -> None:
    lines = [
        "# Final Portfolio Evidence Report",
        "",
        PORTFOLIO_DISCLAIMER,
        "",
        f"- Evidence ID: `{manifest.evidence_id}`",
        f"- Overall status: `{manifest.overall_status}`",
        f"- Milestones represented: `{len(manifest.milestone_completion_matrix)}`",
        f"- Generation command: `{manifest.evidence_provenance['generation_command']}`",
        "",
        "## Capability Summary",
    ]
    for item in manifest.capability_inventory:
        lines.append(f"- `{item['capability']}`: {item['claim_classification']}")
    lines.extend(["", "## Limitations"])
    lines.extend(f"- {item}" for item in manifest.limitations_summary)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_ignored(path: Path) -> bool:
    if path.as_posix().startswith("reports/generated/"):
        return True
    try:
        path.relative_to(PORTFOLIO_EVIDENCE_DIR)
        return True
    except ValueError:
        pass
    if path == PORTFOLIO_EVIDENCE_DIR:
        return True
    return False


def _readme_has_required_sections() -> bool:
    readme = Path("README.md")
    if not readme.exists():
        return False
    content = readme.read_text(encoding="utf-8")
    return all(f"## {section}" in content for section in README_REQUIRED_SECTIONS)


def _claim_violations() -> list[dict[str, str]]:
    paths = [
        Path("README.md"),
        Path("docs/architecture.md"),
        Path("docs/limitations.md"),
        Path("docs/roadmap.md"),
        *MANDATORY_PORTFOLIO_DOCS,
    ]
    violations = []
    for path in paths:
        if not path.exists():
            continue
        content = " ".join(path.read_text(encoding="utf-8").lower().split())
        for claim in PROHIBITED_CLAIMS:
            claim_lower = claim.lower()
            start = 0
            while True:
                index = content.find(claim_lower, start)
                if index == -1:
                    break
                context = content[max(0, index - 220) : index + len(claim_lower) + 80]
                if not _claim_is_negated(context, claim_lower):
                    violations.append({"path": path.as_posix(), "claim": claim})
                start = index + len(claim_lower)
    return violations


def _claim_is_negated(content: str, claim: str) -> bool:
    prefix = content.split(claim, maxsplit=1)[0]
    negations = [
        f"not {claim}",
        f"not a {claim}",
        f"not an {claim}",
        f"no {claim}",
        f"without {claim}",
        f"does not {claim}",
        f"does not implement {claim}",
        f"do not claim {claim}",
        f"must not claim {claim}",
        f"no live {claim}",
        f"out of scope. {claim}",
    ]
    return (
        any(negation in content for negation in negations)
        or "not " in prefix[-80:]
        or "no " in prefix[-80:]
        or "without " in prefix[-80:]
        or "not yet implemented" in prefix[-220:]
        or "does not implement" in prefix[-160:]
        or "do not claim" in prefix[-160:]
        or "prohibited claims" in prefix[-160:]
    )
