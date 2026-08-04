# Enterprise Medical Imaging AI Platform

> This platform is a research and engineering demonstrator. Outputs are intended for technical evaluation and human review only and must not be used for clinical diagnosis or patient-management decisions.

Enterprise Medical Imaging AI Platform is a planned production-oriented medical-imaging AI platform for abdominal CT research workflows. The long-term goal is to demonstrate how DICOM ingestion, de-identification, quality control, image standardisation, longitudinal registration, adrenal-region localisation, lesion analysis, governed review, MLOps, cloud architecture, and clinical AI assurance fit together as one engineered system.

Milestones 1-13 establish repository foundations, synthetic data, DICOM ingestion, quality control,
preprocessing, registration, localisation, synthetic segmentation, and binary synthetic
lesion-presence classification with calibration, governed synthetic longitudinal lesion-change
analysis, a governed local FastAPI research interface, a local Streamlit reviewer UI, and local
container release-assurance controls. Cloud deployment, monitoring, advanced classification, and
clinical deployment are **Planned - not yet implemented**.

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

Complete in Milestone 2:

- Synthetic CT-like engineering fixture generation.
- Synthetic body, adrenal-placeholder, and lesion masks.
- Longitudinal previous/current synthetic pairs.
- Deterministic manifests with SHA-256 checksums.
- Dataset validation and subject-level split validation.
- Synthetic-data governance documentation.

Complete in Milestone 3:

- Synthetic DICOM CT fixture generation for tests and local engineering validation.
- Recursive local DICOM series discovery and grouping.
- Safe technical metadata extraction.
- Deterministic slice ordering.
- Basic structural validation findings.
- Metadata de-identification with private-tag removal, UID remapping, and audit records.

Complete in Milestone 4:

- Technical DICOM quality-control rule catalogue.
- Slice completeness and duplicate-slice checks.
- Metadata consistency checks.
- Pixel-array integrity checks for local fixtures and de-identified data.
- Transparent quality scoring and deterministic JSON/Markdown reports.

Complete in Milestone 5:

- DICOM-series-to-3D NumPy volume assembly for one selected synthetic/de-identified series.
- Internal `[z, y, x]` axis convention with `[z_mm, y_mm, x_mm]` spacing metadata.
- Per-slice rescale slope/intercept conversion to CT-like rescaled intensities.
- Configurable intensity clipping/windowing and normalisation.
- Deterministic engineering crop and padding operations with transform provenance.
- Preprocessing metadata, checksums, Markdown reports, CLI commands, and validation.

Complete in Milestone 6:

- Explicit fixed/moving preprocessed-volume registration workflow.
- SimpleITK centre-of-mass, rigid, and rigid-then-affine registration baselines.
- NumPy/SimpleITK geometry conversion between `[z, y, x]` and `[x, y, z]` conventions.
- Registered moving-volume export, transform JSON, metrics JSON, review arrays, and reports.
- Registration quality gates for input validity, transform plausibility, metrics, and padding.
- Synthetic registration fixture bridge for clean-checkout verification.

Complete in Milestone 7:

- Deterministic atlas-style adrenal-region placeholder localisation.
- Synthetic localisation fixtures with separate left/right binary engineering masks.
- Separate left and right ROI extraction, padding metadata, overlay arrays, and checksums.
- Optional synthetic-label metrics for centre distance, target coverage, IoU, and swap detection.
- Localisation quality gates, reports, CLI commands, and validation.

Complete in Milestone 8:

- Synthetic segmentation dataset preparation from existing synthetic manifests and splits.
- CPU-compatible PyTorch/MONAI 3D U-Net baseline.
- Training, validation, best/last checkpoint export, and deterministic experiment evidence.
- Segmentation inference with sigmoid probability maps, thresholding, and post-processing.
- Dice, IoU, precision, recall, specificity, volume-error, and surface-distance metrics.
- Segmentation model-quality gates, CLI commands, Make targets, and model card.

Complete in Milestone 9:

- Synthetic classification dataset preparation from ROI-like adrenal-side crops.
- Binary labels for `no_visible_synthetic_lesion` and `synthetic_lesion_present`.
- CPU-compatible original PyTorch 3D CNN baseline with no pretrained weights.
- Validation-only calibration and threshold policy evidence.
- Inference abstention with the inference-only `indeterminate` label.
- AUROC, AUPRC, recall, specificity, precision, NPV, Brier score, calibration, and confusion metrics.
- Classification quality gates, CLI commands, Make targets, and model card.

Complete in Milestone 10:

- Previous/current synthetic lesion-mask measurements with spacing-aware physical volumes and diameters.
- Deterministic lesion component matching with centroid, overlap, IoU, and ambiguity handling.
- Absolute and percentage volume and diameter change calculations.
- Engineering labels: `new`, `increased`, `stable`, `reduced`, `resolved`, and `indeterminate`.
- Upstream registration, segmentation, localisation, classification, calibration, and abstention propagation.
- Longitudinal quality gates, evidence exports, review arrays, CLI commands, and Make targets.

Complete in Milestone 11:

- Local governed FastAPI application factory and typed API configuration.
- Health, version, and readiness endpoints with deterministic quality findings.
- Research-only segmentation, classification, and longitudinal API routes backed by existing local pipelines.
- Read-only review evidence endpoints for governed segmentation, classification, and longitudinal outputs.
- Request IDs, request-size limits, safe filesystem root controls, symlink blocking, NumPy `allow_pickle=False`, sanitized errors, and security headers.
- API CLI commands, Make targets, tests, and API design/contract documentation.

Complete in Milestone 12:

- Local Streamlit reviewer UI foundation that consumes governed API contracts only.
- Overview, segmentation, classification, longitudinal, evidence, and governance review pages.
- Typed reviewer UI config, API client, upload security, response formatting, session-state helpers, human-review decisions, and local review export.
- CLI commands, Make targets, tests, documentation, and governance updates for the reviewer workflow.

Planned - not yet implemented:

- Deformable registration, learned localisation beyond the baseline, advanced segmentation,
  benign-versus-malignant or clinical classification, and RECIST or treatment-response assessment.
- MLflow, Docker, Kubernetes, Terraform, and AWS integrations.
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

- PyTorch and MONAI are used for the Milestone 8 synthetic segmentation baseline.
- PyTorch and scikit-learn are used for the Milestone 9 synthetic classification, calibration, and
  thresholding baseline.
- NumPy and SciPy are used for the Milestone 10 synthetic longitudinal measurement and component
  matching baseline.
- FastAPI and Uvicorn are used for the Milestone 11 local governed research API.
- `httpx` is used for API test-client support.
- Streamlit is used for the Milestone 12 local reviewer UI foundation.

Planned - not yet implemented:

- nibabel and OpenCV imaging workflows.
- MLflow, Streamlit, Docker, Kubernetes, and AWS.

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

Synthetic data commands:

```bash
make generate-synthetic-data
make validate-dataset
make verify-synthetic-data
medical-imaging-platform summarise-dataset data/synthetic/generated
```

Generated synthetic arrays are ignored by Git. They are engineering fixtures only and are not clinically realistic CT scans.

DICOM fixture commands:

```bash
make generate-dicom-fixtures
make discover-dicom
make validate-dicom-fixtures
make verify-dicom-ingestion
make verify-dicom-quality
```

Generated DICOM fixtures, de-identified outputs, and audit artefacts are ignored by Git.

Quality reports are engineering data-quality artefacts only; they do not indicate diagnostic adequacy.

Preprocessing commands:

```bash
make preprocess-dicom
make validate-preprocessed-volume
make verify-preprocessing
medical-imaging-platform inspect-preprocessed-volume data/processed/preprocessing/<run_id>
```

Generated preprocessed NumPy volumes and reports are ignored by Git. They are technical artefacts for
research engineering only and do not perform registration, localisation, segmentation,
classification, NIfTI conversion, or spatial resampling.

Registration commands:

```bash
make register-synthetic-pair
make validate-registration
make verify-registration
medical-imaging-platform register-volumes --fixed <fixed-preprocessed-dir> --moving <moving-preprocessed-dir>
```

Registration is technical alignment for research evaluation only. Optimiser convergence and metric
improvement do not prove anatomical correctness or clinical suitability.

Localisation commands:

```bash
make generate-localisation-fixtures
make localise-synthetic-regions
make validate-localisation
make verify-localisation
medical-imaging-platform localise-adrenal-regions <preprocessed-or-registered-dir>
```

Localisation is an atlas/geometry baseline only. Synthetic adrenal-region masks are engineering
placeholders, not clinical annotations, and localisation outputs must not be used diagnostically.

Segmentation commands:

```bash
make prepare-segmentation-data
make train-segmentation
make evaluate-segmentation
make verify-segmentation
medical-imaging-platform train-segmentation ml/datasets/segmentation/<dataset_id>
```

Segmentation uses synthetic engineering lesion masks only. Generated checkpoints under
`ml/experiments/` are ignored by Git and must not be interpreted as clinical-performance evidence.

Classification commands:

```bash
make prepare-classification-data
make train-classification
make evaluate-classification
make verify-classification
medical-imaging-platform train-classification ml/datasets/classification/<dataset_id>
```

Classification uses synthetic ROI-like crops and binary synthetic lesion-presence labels only.
Generated checkpoints, calibration artefacts, threshold policies, and inference outputs under
`ml/experiments/` are ignored by Git and must not be interpreted as clinical-performance evidence.

Longitudinal analysis commands:

```bash
make analyse-synthetic-longitudinal
make validate-longitudinal
make verify-longitudinal
medical-imaging-platform analyse-longitudinal-pair --previous-mask <prev.npy> --current-mask <curr.npy> --previous-spacing 2.5 2.5 2.5 --current-spacing 2.5 2.5 2.5 --case-id <case> --research-subject-id <subject> --side left --registration-run-id <run>
```

Longitudinal labels are synthetic engineering labels only. They are not RECIST, disease progression,
treatment response, diagnosis, or clinical decision support.

API commands:

```bash
make validate-api
make test-api
make verify-api
medical-imaging-platform validate-api-config
medical-imaging-platform inspect-api-readiness
medical-imaging-platform serve-api
```

The API is local-first and research-only. It accepts explicit `.npy` arrays or compact JSON arrays
from configured local roots, returns bounded summaries rather than image volumes, and exposes
read-only evidence review. It does not implement authentication, a dashboard, PACS/DICOMweb
connectivity, databases, queues, cloud deployment, or clinical decision support.

Reviewer UI commands:

```bash
make validate-reviewer-ui
make test-reviewer-ui
make verify-reviewer-ui
medical-imaging-platform validate-reviewer-ui-config
medical-imaging-platform inspect-reviewer-ui-readiness
medical-imaging-platform serve-reviewer-ui
```

The reviewer UI is local-only by default and depends on the governed API. It supports bounded
synthetic review inputs, read-only evidence inspection, explicit human-review decisions, and local
review-session export. It does not execute model inference directly, authenticate users, persist to a
database, or integrate with clinical workflow systems.

## Data Safety

Only synthetic data or publicly available de-identified data may be used. Do not commit real patient data, model weights derived from restricted patient data, credentials, cloud account identifiers, DICOM studies with identifying metadata, or screenshots containing protected health information.

## Research-Only Intended Use

This repository is not an approved medical device, is not validated for NHS deployment, and must not be used for diagnosis or patient-management decisions. Human review is mandatory for any future outputs.

## 16-Milestone Roadmap

1. Repository foundation.
2. Synthetic and public-data foundation. Complete for synthetic-data foundation and public-data selection criteria only; no public data is downloaded.
3. DICOM ingestion and governance. Complete for local synthetic fixtures and metadata de-identification only.
4. Imaging quality control. Complete for technical DICOM engineering checks only.
5. Preprocessing. Complete for deterministic NumPy CT preprocessing foundation only.
6. Registration. Complete for SimpleITK centre-of-mass, rigid, and affine baselines only.
7. Baseline localisation. Complete for atlas-style adrenal-region placeholder localisation only; no segmentation.
8. Synthetic segmentation baseline. Complete for small MONAI 3D U-Net on synthetic masks only.
9. Classification and calibration. Complete for binary synthetic lesion-presence classification only.
10. Longitudinal analysis. Complete for governed synthetic engineering labels only.
11. API foundation. Complete for governed local FastAPI service only; review dashboard remains planned.
12. Reviewer UI foundation. Complete for local Streamlit API-consuming reviewer interface only.
13. MLOps platform.
14. Docker and Kubernetes.
15. AWS deployment blueprint.
16. Clinical AI assurance.
17. Final portfolio packaging.

## Attribution

The project was informed by research review of earlier CT alignment and adrenal lesion detection repositories. This repository is a clean-room original implementation and does not copy their code, notebooks, documentation, repository structures, images, reports, model artefacts, or performance claims. See [docs/attribution.md](docs/attribution.md) and [NOTICE.md](NOTICE.md).

## Limitations

The current repository serves local research API summaries and a local reviewer UI only. It does not perform clinical diagnosis, implement RECIST, provide production monitoring, or
deploy infrastructure. Current modelling and analysis are limited to synthetic segmentation, binary
synthetic lesion-presence classification, and synthetic longitudinal engineering labels. See
[docs/limitations.md](docs/limitations.md) and [governance/limitations.md](governance/limitations.md).
