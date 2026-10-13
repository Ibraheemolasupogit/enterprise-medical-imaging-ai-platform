# Project Summary

This platform is a research and engineering demonstrator. Outputs are intended for technical
evaluation and human review only and must not be used for clinical diagnosis or patient-management
decisions.

Enterprise Medical Imaging AI Platform is a medical-imaging AI engineering demonstrator
for abdominal CT research workflows. It demonstrates synthetic CT generation, governed DICOM
ingestion and de-identification, technical image quality control, preprocessing, longitudinal
registration, adrenal-region localisation, synthetic lesion segmentation, calibrated synthetic
classification, longitudinal lesion analysis, governed API/UI review, local container release
assurance, model registry and monitoring evidence, Kubernetes/Helm assurance, AWS target-state
infrastructure, observability, resilience, incident response and clinical-AI governance boundaries.

The repository is intentionally conservative: it uses synthetic or public de-identified data only,
does not claim diagnostic performance, does not claim NHS or medical-device approval, does not
deploy AWS resources, and does not automate model promotion, retraining or rollback.

## Repository Description

Research-only medical-imaging AI engineering platform demonstrating governed CT data pipelines,
PyTorch/MONAI modelling, API/UI review, MLOps evidence, secure deployment assurance and cloud
target-state architecture using synthetic data.

## M23 consolidated status

The current platform includes M19 classical preprocessing/QC, M20 CNN/ViT classification
comparison, M21 Faster R-CNN/Tiny DETR detection, M22 embeddings/exact retrieval, and M23
interface/evidence consolidation. Historical milestone descriptions above describe their original
scope. The authoritative current flow is [the consolidated architecture](../architecture/final_end_to_end_architecture.md)
and [the canonical demo](../platform_demo.md). Model datasets retain reference-derived labels;
the canonical longitudinal demonstration consumes predictions and preserves quality failures.
Capability status distinguishes implemented, executed/validated, simulated, static design and target state.
