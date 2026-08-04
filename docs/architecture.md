# Architecture

Milestones 1-13 provide repository foundation, synthetic fixtures, local DICOM ingestion,
metadata-focused de-identification, technical DICOM quality control, and deterministic CT
preprocessing to NumPy volumes, SimpleITK registration baselines, and deterministic adrenal-region
placeholder localisation, plus small synthetic lesion segmentation and lesion-presence
classification baselines, governed synthetic longitudinal lesion-change analysis, a local
FastAPI research interface, a local Streamlit reviewer UI, and local container release assurance.

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

Planned - not yet implemented:

- DICOM-to-NIfTI conversion.
- Spatial resampling and full anatomical reorientation.
- Deformable or anatomy-constrained longitudinal registration.
- Learned or anatomy-aware adrenal ROI localisation.
- Advanced lesion segmentation and clinical classification.
- Clinical longitudinal change measurement, RECIST, and treatment-response assessment.
- Production-grade human-review dashboard with authentication and persistent audit logging.
- MLOps monitoring, audit, registry, retraining workflows, and cloud deployment.

The intended architecture will be added incrementally by milestone so each component has tests,
documentation, and governance boundaries before adjacent capabilities depend on it.
