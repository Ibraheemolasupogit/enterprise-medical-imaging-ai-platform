# Data

This repository must not contain real patient data.

Allowed future data categories:

- Synthetic data.
- Publicly available de-identified data with documented provenance and permitted use.

Do not commit DICOM studies, NIfTI volumes, derived patient artefacts, credentials, restricted labels, or model weights.

Milestone 2 generated synthetic arrays live under `data/synthetic/` and are ignored by Git except documentation.

Milestone 3 generated DICOM fixtures, de-identified outputs, and audit artefacts live under `data/dicom/` and are ignored by Git except documentation.
