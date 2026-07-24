# Architecture

Milestones 1-6 provide repository foundation, synthetic fixtures, local DICOM ingestion,
metadata-focused de-identification, technical DICOM quality control, and deterministic CT
preprocessing to NumPy volumes, plus SimpleITK registration baselines.

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

Planned - not yet implemented:

- DICOM-to-NIfTI conversion.
- Spatial resampling and full anatomical reorientation.
- Deformable or anatomy-constrained longitudinal registration.
- Adrenal ROI localisation.
- Lesion segmentation and classification.
- Longitudinal change measurement.
- FastAPI review service.
- Review dashboard.
- Monitoring, audit, registry, and retraining workflows.

The intended architecture will be added incrementally by milestone so each component has tests, documentation, and governance boundaries before adjacent capabilities depend on it.
