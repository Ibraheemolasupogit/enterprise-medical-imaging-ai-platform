# Architecture

Milestone 1 provides only repository foundation, configuration validation, logging, documentation, governance files, tests, and CI.

Planned - not yet implemented:

- DICOM ingestion gateway.
- De-identification and metadata validation.
- Imaging quality-control engine.
- DICOM-to-NIfTI conversion.
- Preprocessing and standardisation.
- Longitudinal registration.
- Adrenal ROI localisation.
- Lesion segmentation and classification.
- Longitudinal change measurement.
- FastAPI review service.
- Review dashboard.
- Monitoring, audit, registry, and retraining workflows.

The intended architecture will be added incrementally by milestone so each component has tests, documentation, and governance boundaries before adjacent capabilities depend on it.
