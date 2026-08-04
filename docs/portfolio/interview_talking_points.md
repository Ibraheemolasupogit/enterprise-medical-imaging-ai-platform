# Interview Talking Points

This platform is a research and engineering demonstrator. Outputs are intended for technical
evaluation and human review only and must not be used for clinical diagnosis or patient-management
decisions.

## 30-Second Summary

I built a medical-imaging AI engineering platform that demonstrates the full lifecycle around CT AI:
synthetic data, DICOM governance, quality control, preprocessing, registration, localisation,
segmentation, calibrated classification, longitudinal analysis, governed API/UI review, model
registry, monitoring, containers, Kubernetes, AWS target architecture and incident-response
evidence. It is deliberately research-only and uses synthetic or public de-identified data only.

## 2-Minute Technical Walkthrough

The repository starts with deterministic synthetic CT-like fixtures, then exercises a governed DICOM
path with metadata de-identification and quality control. Imaging outputs move through
preprocessing, SimpleITK registration and atlas-style adrenal localisation. PyTorch/MONAI baselines
cover segmentation, while a PyTorch classifier demonstrates calibration and abstention. The outputs
feed longitudinal lesion-change analysis, a FastAPI research API and a Streamlit reviewer UI. Later
milestones add CPU-only containers, release evidence, a governed model registry, synthetic
monitoring/drift/audit evidence, secure Helm/Kubernetes manifests, AWS Terraform target-state
architecture, and operations evidence for metrics, SLOs, incidents, rollback and recovery.

## 5-Minute End-To-End Walkthrough

1. Show the README architecture and final Mermaid diagram.
2. Run `make demo-fast` or open `reports/generated/portfolio/portfolio_report.md`.
3. Explain synthetic data and DICOM governance boundaries.
4. Walk through model evidence, calibration and abstention.
5. Show API/UI, release, Kubernetes, AWS and operations evidence.
6. Close with limitations: no clinical use, no NHS approval, no medical-device claims, no live AWS
   deployment, and no automated promotion, retraining or rollback.

## Likely Questions

### Why MONAI and PyTorch?

MONAI provides medical-imaging-oriented transforms and architectures while PyTorch keeps the training
and inference code familiar, testable and CPU-compatible for a portfolio environment.

### Why synthetic data?

Synthetic fixtures make the workflow deterministic, reproducible and safe to share. They avoid
patient-data governance risk but cannot support clinical-performance claims.

### How is calibration handled?

The classification milestone separates validation-only calibration and threshold policy from
inference. Abstention is explicit and must remain visible to reviewers.

### What would be required for clinical validation?

A regulated-quality data strategy, clinical protocol, representative de-identified datasets,
independent validation, usability and safety engineering, regulatory review, security operations,
clinical governance, monitoring and post-market processes would be required. This repository does
not provide those.
