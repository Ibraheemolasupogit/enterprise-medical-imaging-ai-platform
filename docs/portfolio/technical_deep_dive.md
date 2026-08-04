# Technical Deep Dive

This platform is a research and engineering demonstrator. Outputs are intended for technical
evaluation and human review only and must not be used for clinical diagnosis or patient-management
decisions.

## Architecture Decisions

- Python 3.12 keeps the implementation current and straightforward to validate.
- PyTorch and MONAI are used for modelling because they are common in medical-imaging research and
  support CPU-compatible baselines.
- FastAPI and Streamlit separate governed API contracts from the reviewer workflow.
- Docker and Helm demonstrate deployment hardening without requiring a live production cluster.
- Terraform captures AWS target-state architecture while keeping `terraform apply` out of scope.

## Data Strategy

The data path is synthetic-first. Synthetic CT-like fixtures, synthetic DICOM, masks and labels make
the repository reproducible and safe to share. This avoids patient-data leakage but means model
metrics are engineering signals only.

## Model Development And Evaluation

Segmentation uses a compact MONAI 3D U-Net baseline. Classification uses a compact PyTorch 3D CNN
with calibration and threshold-policy evidence. Both are synthetic baselines with model cards,
quality gates and checkpoint checksums. They do not claim clinical lesion-detection performance.

## Governance And Human Oversight

Model registry transitions require explicit human approval metadata. Reviewer UI decisions are
separate from model outputs. Monitoring alerts, SLO breaches and incident simulations require human
investigation and documented change control.

## Deployment Assurance

Container evidence checks CPU-only dependencies, non-root users, read-only root filesystems,
scanners, SBOMs and smoke tests. Kubernetes evidence checks Helm rendering, secure pod settings,
NetworkPolicies, probes and optional local kind runtime. AWS evidence maps private ECR/EKS/S3/KMS
and observability boundaries without deploying resources.

## Trade-Offs

The project favours deterministic evidence, safety boundaries and reviewability over clinical
realism. It intentionally avoids real patient data, cloud deployment, automatic promotion,
automatic retraining and automatic rollback.
