# Limitations

Current limitations:

- Synthetic CT-like engineering fixtures exist, but real patient data must not be loaded.
- DICOM ingestion is limited to local-file discovery, metadata extraction, synthetic fixtures, structural validation, and metadata de-identification.
- No PACS, DICOMweb, NHS integration, or network DICOM workflow exists.
- Metadata de-identification cannot guarantee removal of burned-in pixel identifiers.
- Preprocessing is limited to deterministic NumPy assembly, technical rescale conversion, intensity transforms, engineering crop/pad operations, metadata, and validation.
- Registration is limited to SimpleITK centre-of-mass, rigid, and affine engineering baselines for preprocessed volumes.
- Localisation is limited to deterministic atlas-style left/right adrenal-region placeholders for synthetic engineering validation.
- Segmentation is limited to a small PyTorch/MONAI 3D U-Net trained on synthetic engineering lesion masks.
- Classification is limited to a small PyTorch 3D CNN trained on synthetic ROI-like crops for binary synthetic lesion presence only.
- Longitudinal analysis is limited to synthetic mask measurements, deterministic matching, and engineering change labels.
- No NIfTI export, general preprocessing resampling, full anatomical reorientation, deformable registration, learned localisation beyond the baseline, advanced segmentation, benign-versus-malignant classification, RECIST, treatment-response assessment, or clinical diagnosis exists.
- A local governed FastAPI research API, Streamlit reviewer UI, local container release evidence,
  local synthetic registry/monitoring/audit evidence, and secure Helm/Kubernetes chart evidence
  exist, but no authentication, production audit store, MLflow, Terraform, AWS, EKS, service mesh, or
  production cloud implementation exists.
- No clinical validation has been performed.
- Synthetic volumes are not clinically realistic CT scans.
- Synthetic lesion shapes do not represent the diversity of real pathology.
- Synthetic data cannot support clinical-performance claims.
- Milestone 3 does not establish regulatory compliance or full DICOM confidentiality-profile compliance.
- Milestone 4 quality scores are engineering indicators only and do not establish diagnostic adequacy.
- Milestone 5 preprocessing reports are engineering provenance only and do not establish diagnostic image standardisation.
- Milestone 6 registration reports are engineering provenance only. Optimiser convergence, metric improvement, and synthetic translation recovery do not establish anatomical correctness.
- Milestone 7 localisation reports are engineering provenance only. Synthetic placeholder masks and
  atlas centres do not establish adrenal localisation accuracy or anatomical correctness.
- Milestone 8 segmentation reports are engineering evidence only. Synthetic Dice, recall, volume
  error, and surface distances do not establish clinical lesion segmentation performance.
- Milestone 9 classification reports are engineering evidence only. Synthetic AUROC, AUPRC, recall,
  Brier score, calibration diagnostics, threshold behavior, and abstention rates do not establish
  clinical lesion detection or diagnostic classification performance.
- Milestone 10 longitudinal reports are engineering evidence only. Synthetic `new`, `increased`,
  `stable`, `reduced`, `resolved`, and `indeterminate` labels do not establish progression,
  response, RECIST category, diagnosis, or patient-management suitability.
- Milestone 11 API responses are engineering summaries only. API availability, readiness, checksums,
  and successful requests do not establish clinical safety, diagnostic performance, or deployment
  approval.
- Milestone 12 reviewer decisions are engineering workflow artefacts only. They are not clinical
  approval, diagnosis, treatment-response assessment, RECIST compliance, or medical-device evidence.
- Milestone 14 registry, monitoring, drift, alert, and audit evidence is deterministic local
  engineering evidence only. Drift findings do not establish clinical performance deterioration,
  model safety, deployment readiness, NHS approval, or medical-device compliance. No automated
  retraining, rollback, deployment, or model promotion is implemented.
- Milestone 15 Kubernetes and Helm evidence is local deployment assurance only. Static policy checks
  do not establish production security, cluster admission behavior, cloud readiness, high
  availability, clinical operational approval, or medical-device compliance. Runtime smoke is
  recorded as `UNAVAILABLE` when local tooling is absent and must not be represented as a pass.
- Milestone 16 AWS and Terraform evidence is target-state infrastructure evidence only. It does not
  deploy AWS resources, run `terraform apply`, publish images, configure production DNS, install real
  TLS certificates, establish high availability, prove cloud operational readiness, or create a live
  clinical service.
- Milestone 17 observability, SLO, incident, rollback and recovery evidence is deterministic local
  engineering evidence only. It does not create a production monitoring stack, send real alerts,
  page operators, prove high availability, satisfy clinical safety surveillance, deploy AWS
  resources, run Terraform apply, automate rollback, automate retraining, or promote model versions.
- Missing slices are inferred only when spacing evidence is reliable; metadata may be incomplete or inaccurate.

The platform is a portfolio and research demonstrator, not an approved medical device.
## Container Release Limitations

Milestone 13 container images and release evidence are local engineering artefacts only. They do not
represent production deployment readiness, internet-facing security, clinical operational approval,
cloud certification, high availability, disaster recovery, or Kubernetes readiness.

## Kubernetes Deployment Limitations

Milestone 15 Helm and Kubernetes artefacts are local engineering deployment evidence only. They do
not implement AWS, EKS, Terraform, GPU scheduling, service mesh, production ingress, production
identity, automated rollback, automated retraining, automatic model promotion, or clinical
deployment. Inference endpoints remain internal by default.

## AWS Infrastructure Limitations

Milestone 16 Terraform artefacts are static, validation-safe design artefacts. They do not create
paid resources unless an operator separately chooses to run Terraform outside the repository's normal
checks. No AWS credentials are required for evidence generation, and missing optional scanners are
recorded as `UNAVAILABLE` rather than treated as security approval.

## Operations Limitations

Milestone 17 operations artefacts validate local control design and deterministic evidence
generation only. Prometheus metrics are opt-in and local, CloudWatch resources remain target-state
Terraform definitions only, and incident simulations are scripted engineering scenarios rather than
live operational incidents. All rollback and recovery decisions remain manual change-control
activities.
