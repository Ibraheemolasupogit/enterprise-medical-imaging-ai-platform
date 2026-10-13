"""Lightweight repository documentation validation."""

from __future__ import annotations

import re
from pathlib import Path

REQUIRED_MARKDOWN_FILES = (
    Path("README.md"),
    Path("docs/README.md"),
    Path("api/README.md"),
    Path("dashboard/README.md"),
    Path("ml/README.md"),
    Path("infrastructure/README.md"),
    Path("monitoring/README.md"),
    Path("docs/detection_design.md"),
    Path("docs/embeddings.md"),
    Path("docs/platform_demo.md"),
    Path("docs/local_startup.md"),
    Path("docs/evidence_contracts.md"),
    Path("docs/engineering_comparison.md"),
    Path("NOTICE.md"),
    Path("CONTRIBUTING.md"),
    Path("SECURITY.md"),
    Path("docs/api_design.md"),
    Path("docs/api_contracts.md"),
    Path("docs/architecture.md"),
    Path("docs/architecture/final_end_to_end_architecture.md"),
    Path("docs/clinical_workflow.md"),
    Path("docs/classification_design.md"),
    Path("docs/containerisation.md"),
    Path("docs/data_flow.md"),
    Path("docs/data_provenance.md"),
    Path("docs/deidentification_design.md"),
    Path("docs/deployment/aws_architecture.md"),
    Path("docs/deployment/disaster_recovery.md"),
    Path("docs/deployment/kubernetes.md"),
    Path("docs/deployment/kubernetes_operations.md"),
    Path("docs/deployment/terraform_operations.md"),
    Path("docs/dicom_ingestion_design.md"),
    Path("docs/dicom_metadata_model.md"),
    Path("docs/dataset_selection_criteria.md"),
    Path("docs/imaging_quality_control.md"),
    Path("docs/localisation_design.md"),
    Path("docs/local_release_assurance.md"),
    Path("docs/longitudinal_analysis_design.md"),
    Path("docs/operations/incident_response.md"),
    Path("docs/operations/observability.md"),
    Path("docs/operations/rollback_recovery.md"),
    Path("docs/operations/runbooks.md"),
    Path("docs/operations/slo_error_budget.md"),
    Path("docs/portfolio/project_summary.md"),
    Path("docs/portfolio/demonstration_script.md"),
    Path("docs/portfolio/technical_deep_dive.md"),
    Path("docs/preprocessing_baseline.md"),
    Path("docs/roadmap.md"),
    Path("docs/registration_design.md"),
    Path("docs/release_evidence.md"),
    Path("docs/reviewer_ui_design.md"),
    Path("docs/reviewer_workflow.md"),
    Path("docs/segmentation_design.md"),
    Path("docs/security.md"),
    Path("docs/synthetic_data_design.md"),
    Path("docs/attribution.md"),
    Path("docs/limitations.md"),
    Path("governance/intended_use.md"),
    Path("governance/excluded_use.md"),
    Path("governance/human_oversight.md"),
    Path("governance/clinical_safety_approach.md"),
    Path("governance/data_protection_design.md"),
    Path("governance/model_change_control.md"),
    Path("governance/monitoring_plan.md"),
    Path("governance/limitations.md"),
    Path("ml/dataset_cards/synthetic_ct_dataset.md"),
    Path("ml/model_cards/segmentation_baseline.md"),
    Path("ml/model_cards/classification_baseline.md"),
)

DISCLAIMER = (
    "This platform is a research and engineering demonstrator. Outputs are intended "
    "for technical evaluation and human review only and must not be used for clinical "
    "diagnosis or patient-management decisions."
)

README_REQUIRED_SECTIONS = (
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
)

PROHIBITED_UNQUALIFIED_CLAIMS = (
    "nHS approved",
    "live clinical deployment",
    "autonomous clinical decisions",
    "clinical decision support system",
)


def validate_markdown_file(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"Missing required documentation file: {path}"]

    content = path.read_text(encoding="utf-8")
    if not content.strip():
        errors.append(f"Documentation file is empty: {path}")
    if "\t" in content:
        errors.append(f"Documentation file contains tab characters: {path}")
    if not content.endswith("\n"):
        errors.append(f"Documentation file must end with a newline: {path}")
    if path.suffix == ".md" and not content.lstrip().startswith("#"):
        errors.append(f"Markdown file must start with a heading: {path}")
    return errors


def validate_claims(path: Path, content: str) -> list[str]:
    """Check each occurrence in its own sentence/contrast clause, not the whole document."""
    errors = []
    for claim in PROHIBITED_UNQUALIFIED_CLAIMS:
        for match in re.finditer(re.escape(claim), content, flags=re.IGNORECASE):
            preceding = content[max(0, match.start() - 200) : match.start()].lower()
            prefix = re.split(r"[.!?;]|\bbut\b|\bhowever\b", preceding)[-1]
            prefix = prefix.rsplit(claim.lower(), maxsplit=1)[-1]
            if not re.search(
                r"\b(no|not|without|prohibited|must not|do not|does not)\b[^.!?;]*$", prefix
            ):
                errors.append(f"{path} contains unsupported claim wording: {claim}")
    return errors


def validate_current_state(path: Path, content: str) -> list[str]:
    errors = []
    if path.as_posix() in {
        "api/README.md",
        "dashboard/README.md",
        "ml/README.md",
        "infrastructure/README.md",
        "monitoring/README.md",
        "SECURITY.md",
    } and re.search(
        r"planned\s*[-–—]\s*not yet implemented|repository foundation code only",
        content,
        flags=re.IGNORECASE,
    ):
        errors.append(f"{path} contains stale implemented-capability scaffold wording.")
    if path.as_posix() == "docs/roadmap.md":
        milestones = {int(value) for value in re.findall(r"^(\d+)\.", content, re.MULTILINE)}
        if not set(range(19, 24)) <= milestones or "review dashboard remains planned" in content:
            errors.append("docs/roadmap.md does not represent current M19–M23 capabilities.")
    return errors


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED_MARKDOWN_FILES:
        errors.extend(validate_markdown_file(path))

    readme = Path("README.md").read_text(encoding="utf-8")
    if DISCLAIMER not in readme:
        errors.append("README.md is missing the required research-only disclaimer.")
    for section in README_REQUIRED_SECTIONS:
        if f"## {section}" not in readme:
            errors.append(f"README.md is missing required section: {section}")

    roadmap = Path("docs/roadmap.md").read_text(encoding="utf-8")
    if "18. Clinical AI assurance and final portfolio packaging. Planned." in roadmap:
        errors.append("docs/roadmap.md contains stale Milestone 18 placeholder wording.")

    for path in REQUIRED_MARKDOWN_FILES:
        content = path.read_text(encoding="utf-8")
        errors.extend(validate_claims(path, content))
        errors.extend(validate_current_state(path, content))

    if errors:
        for error in errors:
            print(error)
        return 1

    print(f"Validated {len(REQUIRED_MARKDOWN_FILES)} documentation files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
