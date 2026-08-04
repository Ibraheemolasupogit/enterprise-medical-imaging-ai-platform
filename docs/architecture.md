# Architecture

Milestones 1-17 provide repository foundation, synthetic fixtures, local DICOM ingestion,
metadata-focused de-identification, technical DICOM quality control, and deterministic CT
preprocessing to NumPy volumes, SimpleITK registration baselines, and deterministic adrenal-region
placeholder localisation, plus small synthetic lesion segmentation and lesion-presence
classification baselines, governed synthetic longitudinal lesion-change analysis, a local
FastAPI research interface, a local Streamlit reviewer UI, and local container release assurance.
Milestone 14 adds a local governed model registry, deterministic synthetic monitoring/drift
evidence, and append-only JSONL audit evidence. Milestone 15 adds secure Helm/Kubernetes packaging
and checkout-safe static deployment assurance for the API and reviewer UI. Milestone 16 adds a
controlled AWS target-state architecture and Terraform static assurance for ECR, EKS, S3, KMS,
Secrets Manager references, CloudWatch, CloudTrail, VPC networking, IAM boundaries, and cost-aware
validation defaults without deploying AWS resources. Milestone 17 adds opt-in application metrics,
structured log redaction, resilience controls, operations SLOs, incident simulations, rollback and
recovery evidence, and runbook validation without deploying production monitoring infrastructure.

Implemented foundation components:

- Configuration validation, logging, documentation, governance files, tests, and CI.
- Synthetic CT-like engineering fixture generation.
- Local DICOM fixture generation, discovery, safe metadata extraction, ordering, validation, and
  metadata de-identification.
- Technical DICOM quality-control reports.
- CT-like preprocessing with `[z, y, x]` NumPy assembly, geometry provenance, intensity transforms,
  engineering crop/pad transforms, checksums, and validation.
- Longitudinal registration foundation with explicit fixed/moving roles, centre-of-mass, rigid, and
  affine baselines, transform export, metrics, quality gates, and review arrays.
- Atlas-style localisation foundation with separate left/right configured anatomical ROIs,
  deterministic ROI arrays, synthetic engineering-label metrics, quality gates, and reports.
- PyTorch/MONAI segmentation foundation with prepared synthetic samples, a small 3D U-Net,
  deterministic training evidence, inference outputs, metrics, quality gates, and model card.
- PyTorch classification foundation with synthetic ROI-like crops, a compact 3D CNN, validation-only
  calibration and threshold policy, inference abstention, deterministic evidence, quality gates, and
  model card.
- Longitudinal analysis foundation with spacing-aware mask measurements, deterministic component
  matching, synthetic engineering change labels, upstream quality propagation, review arrays,
  evidence checksums, and reports.
- API foundation with local FastAPI app factory, typed API config, health/readiness/version routes,
  governed prediction/analysis routes, read-only evidence review, deterministic API errors, request
  IDs, request limits, and filesystem allowlist controls.
- Reviewer UI foundation with typed config, no direct model execution, governed API client, overview
  status, synthetic review pages, read-only evidence inspection, human-review decisions, session
  state controls, upload validation, and local review export.
- Container release-assurance foundation with separate API and reviewer UI images, Docker Compose
  local orchestration, non-root runtime users, read-only root filesystems, static policy checks,
  optional scanner evidence, smoke-test orchestration, SBOM hooks, and ignored release manifests.
- Governance evidence foundation with local registry manifests for synthetic segmentation and
  classification model versions, explicit human approval metadata, deterministic synthetic
  monitoring baselines/windows, simple drift checks, alert summaries, append-only audit JSONL,
  checksums, and Markdown reports.
- Kubernetes deployment foundation with a Helm chart, API and reviewer UI Deployments, internal
  Services, ConfigMap, optional Secret references, ServiceAccount, NetworkPolicy, HPA, PDB,
  conservative resources, secure pod/container contexts, rendered manifests, static policy checks,
  and ignored evidence under `reports/generated/kubernetes/`.
- AWS infrastructure-as-code foundation with modular Terraform under `infra/terraform/`, private
  ECR repositories, private-by-default EKS design, governed S3 buckets, customer-managed KMS keys,
  explicit Secrets Manager references, CloudWatch/CloudTrail architecture, custom static policy
  checks, cost-driver evidence, and ignored evidence under `reports/generated/aws/`.
- Operations foundation with protected opt-in Prometheus-format API metrics, structured log-event
  redaction, degraded-readiness signalling, bounded reviewer UI retries and circuit breaking, Helm
  observability/resilience settings, CloudWatch target mappings, deterministic SLO/error-budget
  evaluation, incident simulations, rollback/recovery evidence, runbook validation, checksums, and
  ignored evidence under `reports/generated/operations/`.

Planned - not yet implemented:

- DICOM-to-NIfTI conversion.
- Spatial resampling and full anatomical reorientation.
- Deformable or anatomy-constrained longitudinal registration.
- Learned or anatomy-aware adrenal ROI localisation.
- Advanced lesion segmentation and clinical classification.
- Clinical longitudinal change measurement, RECIST, and treatment-response assessment.
- Production-grade human-review dashboard with authentication and persistent audit logging.
- Automated retraining, automated rollback, model auto-promotion, service mesh, GPU scheduling,
  public inference exposure, Terraform apply workflows, live cloud deployment, production alert
  routing, and production incident-management integrations.

The intended architecture will be added incrementally by milestone so each component has tests,
documentation, and governance boundaries before adjacent capabilities depend on it.
