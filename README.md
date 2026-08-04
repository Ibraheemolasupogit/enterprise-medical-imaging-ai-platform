# Enterprise Medical Imaging AI Platform

> This platform is a research and engineering demonstrator. Outputs are intended for technical evaluation and human review only and must not be used for clinical diagnosis or patient-management decisions.

## Project Summary

Enterprise Medical Imaging AI Platform is a portfolio-scale medical-imaging AI engineering
demonstrator for abdominal CT research workflows. It shows how governed imaging data pipelines,
PyTorch/MONAI modelling, human review, MLOps evidence, secure deployment packaging, AWS target-state
architecture, observability and clinical-AI governance boundaries can fit together in one
repository.

The project uses synthetic or public de-identified data only. It is not a diagnostic system, not an
approved medical device, not NHS approved, not a live clinical deployment and not a clinical
decision-making tool.

## Problem Statement

Longitudinal CT review often requires careful comparison of current and previous imaging, technical
quality checks, lesion localisation, model uncertainty handling, human review and strong evidence
provenance. Many prototypes demonstrate one piece of that chain. This repository demonstrates the
engineering structure around the full workflow while keeping clinical claims deliberately out of
scope.

## Architecture Overview

The final architecture is documented in
[docs/architecture/final_end_to_end_architecture.md](docs/architecture/final_end_to_end_architecture.md).

High-level flow:

1. Synthetic CT-like fixture generation.
2. Governed DICOM ingestion, structural validation and metadata de-identification.
3. Technical image quality control.
4. CT preprocessing to internal NumPy volumes.
5. Longitudinal registration and adrenal-region localisation.
6. Synthetic segmentation and calibrated binary synthetic classification.
7. Longitudinal lesion-change engineering analysis.
8. Governed FastAPI research API and Streamlit reviewer UI.
9. Local container release assurance.
10. Model registry, monitoring, drift and audit evidence.
11. Helm/Kubernetes assurance.
12. AWS target-state Terraform evidence.
13. Observability, resilience, incident, rollback and recovery evidence.
14. Final portfolio evidence pack.

## Implemented Capabilities

- Synthetic CT generation and subject-level split validation.
- Synthetic DICOM fixture generation, local discovery, validation and metadata de-identification.
- Technical DICOM quality control and reporting.
- CT-like preprocessing with geometry, intensity, crop/pad provenance and checksums.
- SimpleITK centre-of-mass, rigid and affine registration baselines.
- Deterministic adrenal-region localisation placeholders.
- CPU-compatible PyTorch/MONAI 3D U-Net segmentation baseline.
- CPU-compatible PyTorch classification baseline with calibration, thresholds and abstention.
- Synthetic longitudinal lesion measurement and change-label evidence.
- Governed local FastAPI API and local Streamlit reviewer UI.
- CPU-only Docker container strategy, scanner hooks, SBOM hooks and release evidence.
- Governed model registry, synthetic monitoring, drift simulation and append-only audit evidence.
- Secure Helm chart, Kubernetes policy checks and optional local kind smoke workflow.
- Terraform AWS target architecture for ECR, EKS, S3, KMS, Secrets Manager references, CloudWatch
  and CloudTrail.
- Structured logging, opt-in protected metrics, SLOs, incidents, rollback/recovery simulations and
  operations runbooks.
- Final portfolio matrix, evidence report and interview materials.

## End-To-End Workflow

Canonical local workflow:

```bash
make demo
```

`make demo` is the longer technical path and may take a while on CPU because it exercises synthetic
data, DICOM/QC, preprocessing, registration, localisation, small model workflows, monitoring,
deployment assurance and portfolio evidence.

Short interview workflow:

```bash
make demo-fast
```

`make demo-fast` is deterministic and uses synthetic evidence. It avoids AWS credentials, paid cloud
deployment and persistent Kubernetes requirements.

Final portfolio readiness:

```bash
make portfolio-readiness
```

This is the canonical checkout-safe consolidation target. It runs quality checks, documentation
validation, monitoring evidence, container static checks, Kubernetes static evidence, AWS static
policy checks, operations evidence and portfolio evidence.

## Technology Stack

- Python 3.12.
- NumPy, SciPy, pydicom and SimpleITK for imaging workflows.
- PyTorch and MONAI for synthetic segmentation and classification baselines.
- scikit-learn for classification metrics and calibration support.
- FastAPI, Uvicorn and httpx for API workflows.
- Streamlit for the local reviewer UI.
- Docker and Docker Compose for local containers.
- Helm and Kubernetes manifests for deployment assurance.
- Terraform for AWS target-state infrastructure.
- pytest, coverage, ruff, mypy and Bandit for quality and security checks.

## Security And Governance

Core boundaries:

- Synthetic or public de-identified data only.
- No credentials, patient data, generated DICOM, model checkpoints or evidence artefacts in Git.
- DICOM metadata de-identification and audit evidence.
- API path allowlists, symlink rejection, request limits and sanitized errors.
- Metrics endpoint disabled by default and protected when configured.
- Containers and Kubernetes pods run non-root with read-only root filesystems and dropped
  capabilities.
- Human approval is mandatory for model lifecycle changes.
- No automated model promotion, retraining or rollback.

Governance documentation lives under [governance](governance) and includes intended use, excluded
use, human oversight, monitoring, model change control, limitations and the hazard log.

## MLOps And Monitoring

The repository includes local deterministic evidence for:

- Model registry lifecycle states: `candidate`, `approved`, `rejected`, `retired`.
- Synthetic monitoring baselines and synthetic drift simulation.
- Append-only JSONL audit evidence.
- SLO and error-budget evidence.
- Incident lifecycle simulation.
- Rollback and recovery simulation.

These are engineering governance signals only. They are not clinical performance monitoring,
diagnostic safety surveillance or production post-market monitoring.

## Deployment Assurance

Deployment evidence is split by claim type:

- Containers: locally executed release assurance for CPU-only images and Docker Compose smoke.
- Kubernetes/Helm: static validation plus optional local kind runtime smoke.
- AWS: target-state Terraform architecture only, not deployed.
- Operations: simulated local evidence for observability, resilience and incident response.

Repository validation does not require AWS credentials, paid services, external patient data, live
clinical systems or Terraform apply.

## AWS Target-State Boundary

AWS Terraform under [infra/terraform](infra/terraform) describes a private-by-default target
architecture with ECR, EKS, S3, KMS, Secrets Manager references, CloudWatch and CloudTrail. It is
static infrastructure-as-code evidence only. The repository has no `terraform apply` target and does
not deploy AWS resources.

## Quick Start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
make quality
```

## Demo Commands

```bash
make demo-fast
make build-portfolio-evidence
make validate-portfolio-evidence
make portfolio-readiness
make clean-demo
```

Optional runtime commands are documented in
[docs/portfolio/demonstration_script.md](docs/portfolio/demonstration_script.md).

## Evidence Locations

Generated artefacts are ignored by Git.

- Synthetic data: `data/synthetic/generated/`
- DICOM and QC: `data/dicom/`
- Processed imaging outputs: `data/processed/`
- Model datasets and experiments: `ml/datasets/`, `ml/experiments/`
- Reviewer sessions: `reports/generated/reviewer-sessions/`
- Release evidence: `reports/generated/releases/`
- Registry, monitoring and audit: `reports/generated/registry/`, `reports/generated/monitoring/`,
  `reports/generated/audit/`
- Kubernetes evidence: `reports/generated/kubernetes/`
- AWS evidence: `reports/generated/aws/`
- Operations evidence: `reports/generated/operations/`
- Portfolio evidence: `reports/generated/portfolio/`

## Test And Quality Status

Latest local validation for Milestone 18:

- `make quality`: PASS.
- Tests: 269 passed.
- Coverage: 90.49%.
- Bandit: no issues identified.

Run:

```bash
make quality
git diff --check
```

## Limitations

- Synthetic data cannot support clinical-performance claims.
- No diagnosis, triage, RECIST, treatment-response or patient-management behavior.
- No PACS, DICOMweb, NHS integration, production authentication or production clinical workflow.
- No live AWS deployment, production DNS, production TLS, service mesh or public inference.
- No automated model promotion, retraining or rollback.
- AWS IaC is target-state only unless an operator separately deploys it outside repository
  validation.
- Kubernetes evidence is deployment assurance, not proof of production security or high
  availability.

See [docs/limitations.md](docs/limitations.md).

## Interview Talking Points

Use these files for recruiter and technical-review preparation:

- [docs/portfolio/project_summary.md](docs/portfolio/project_summary.md)
- [docs/portfolio/cv_project_entry.md](docs/portfolio/cv_project_entry.md)
- [docs/portfolio/interview_talking_points.md](docs/portfolio/interview_talking_points.md)
- [docs/portfolio/demonstration_script.md](docs/portfolio/demonstration_script.md)
- [docs/portfolio/technical_deep_dive.md](docs/portfolio/technical_deep_dive.md)
- [docs/portfolio/recruiter_faq.md](docs/portfolio/recruiter_faq.md)

## Attribution

The project was informed by research review of earlier CT alignment and adrenal lesion detection
repositories. This repository is a clean-room original implementation and does not copy their code,
notebooks, documentation, repository structures, images, reports, model artefacts or performance
claims. See [docs/attribution.md](docs/attribution.md) and [NOTICE.md](NOTICE.md).
