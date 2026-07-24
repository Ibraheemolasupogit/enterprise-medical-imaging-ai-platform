# Enterprise Medical Imaging AI Platform

> This platform is a research and engineering demonstrator. Outputs are intended for technical evaluation and human review only and must not be used for clinical diagnosis or patient-management decisions.

Enterprise Medical Imaging AI Platform is a planned production-oriented medical-imaging AI platform for abdominal CT research workflows. The long-term goal is to demonstrate how DICOM ingestion, de-identification, quality control, image standardisation, longitudinal registration, adrenal-region localisation, lesion analysis, governed review, MLOps, cloud architecture, and clinical AI assurance fit together as one engineered system.

Milestone 1 establishes only the repository foundation. Medical-imaging pipelines, models, APIs, dashboards, cloud deployment, containers, and monitoring are **Planned - not yet implemented**.

## Clinical And Engineering Problem

Radiologists often compare current and previous CT scans to understand whether suspected lesions are new, stable, growing, shrinking, or indeterminate. Research prototypes commonly focus on one part of this workflow, such as registration or classification. This repository is intended to grow into an end-to-end demonstrator that treats clinical safety, reproducibility, human review, auditability, and deployment engineering as first-class concerns.

## Planned Workflow

All items below are **Planned - not yet implemented** unless explicitly marked as complete.

1. Ingest synthetic or publicly available de-identified DICOM studies.
2. Validate metadata, slice ordering, completeness, and image quality.
3. Record de-identification and provenance actions.
4. Convert and standardise volumetric CT data.
5. Register longitudinal current and previous studies.
6. Localise adrenal regions of interest.
7. Segment and classify suspected lesions where appropriate.
8. Measure lesion volume and longitudinal change.
9. Generate confidence, uncertainty, quality, and review indicators.
10. Serve results through governed APIs and a human-review interface.
11. Track experiments, model versions, approval gates, audit events, and monitoring signals.

## Current Implementation Status

Complete in Milestone 1:

- Repository structure.
- Python packaging.
- YAML configuration loading and validation.
- Structured JSON-compatible logging.
- Minimal CLI.
- Documentation foundation.
- Governance foundation.
- Tests and code-quality tooling.
- Basic CI.

Planned - not yet implemented:

- DICOM ingestion and de-identification.
- Medical image preprocessing.
- Registration, localisation, segmentation, and classification.
- FastAPI, Streamlit, MLflow, Docker, Kubernetes, Terraform, and AWS integrations.
- Any diagnostic or clinical decision support behavior.

## Architecture

The proposed architecture is documented in [docs/architecture.md](docs/architecture.md). At a high level, future milestones will add independent components for data ingestion, governance, quality control, preprocessing, registration, modelling, longitudinal analysis, review, monitoring, and deployment.

## Technology Stack

Milestone 1 uses a deliberately small Python 3.12 stack:

- Python packaging with `pyproject.toml`.
- `PyYAML` for YAML parsing.
- `pydantic` for typed validation.
- Standard-library `argparse` and `logging` for CLI and logs.
- `pytest`, `ruff`, `mypy`, `bandit`, and coverage tooling for quality.

Planned - not yet implemented:

- PyTorch and MONAI modelling.
- SimpleITK, pydicom, nibabel, and OpenCV imaging workflows.
- MLflow, FastAPI, Streamlit, Docker, Kubernetes, and AWS.

## Local Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Useful commands:

```bash
medical-imaging-platform version
medical-imaging-platform validate-config
make quality
```

## Quality Commands

```bash
make format
make format-check
make lint
make type-check
make validate-config
make validate-docs
make test
make security
make quality
```

`make quality` is the canonical clean-checkout validation target.

## Data Safety

Only synthetic data or publicly available de-identified data may be used. Do not commit real patient data, model weights derived from restricted patient data, credentials, cloud account identifiers, DICOM studies with identifying metadata, or screenshots containing protected health information.

## Research-Only Intended Use

This repository is not an approved medical device, is not validated for NHS deployment, and must not be used for diagnosis or patient-management decisions. Human review is mandatory for any future outputs.

## 16-Milestone Roadmap

1. Repository foundation.
2. Synthetic and public-data foundation.
3. DICOM ingestion and governance.
4. Imaging quality control.
5. Preprocessing.
6. Registration.
7. Baseline localisation and segmentation.
8. Advanced models.
9. Classification and calibration.
10. Longitudinal analysis.
11. API and review dashboard.
12. MLOps platform.
13. Docker and Kubernetes.
14. AWS deployment blueprint.
15. Clinical AI assurance.
16. Final portfolio packaging.

## Attribution

The project was informed by research review of earlier CT alignment and adrenal lesion detection repositories. This repository is a clean-room original implementation and does not copy their code, notebooks, documentation, repository structures, images, reports, model artefacts, or performance claims. See [docs/attribution.md](docs/attribution.md) and [NOTICE.md](NOTICE.md).

## Limitations

The current repository foundation does not process medical images, train models, serve predictions, or deploy infrastructure. See [docs/limitations.md](docs/limitations.md) and [governance/limitations.md](governance/limitations.md).
