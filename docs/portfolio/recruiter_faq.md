# Recruiter FAQ

This platform is a research and engineering demonstrator. Outputs are intended for technical
evaluation and human review only and must not be used for clinical diagnosis or patient-management
decisions.

## What does this project demonstrate?

It demonstrates end-to-end medical-imaging AI engineering: data generation, DICOM governance,
quality control, image processing, PyTorch/MONAI modelling, API/UI review, MLOps evidence,
containers, Kubernetes, AWS target architecture, monitoring, incident response and governance.

## Is it a medical device?

No. It is not a diagnostic system, approved medical device, NHS-approved system or clinical
decision-making tool.

## Does it use patient data?

No. The repository is built around synthetic or public de-identified data only.

## Does it deploy to AWS?

No. Terraform describes target-state AWS architecture and static validation only. Repository
commands do not deploy AWS resources or run Terraform apply.

## What role is this project relevant for?

It is relevant for machine-learning engineering, medical-imaging AI engineering, MLOps, platform
engineering, cloud architecture and responsible-AI governance roles.

## What is the quickest demo?

Run:

```bash
make demo-fast
```

Then open `reports/generated/portfolio/portfolio_report.md`.
