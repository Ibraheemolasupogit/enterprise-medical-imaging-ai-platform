# Data

This repository must not contain real patient data.

Allowed future data categories:

- Synthetic data.
- Publicly available de-identified data with documented provenance and permitted use.

Do not commit DICOM studies, NIfTI volumes, derived patient artefacts, credentials, restricted labels, or model weights.

Milestone 2 generated synthetic arrays live under `data/synthetic/` and are ignored by Git except documentation.

Milestone 3 generated DICOM fixtures, de-identified outputs, and audit artefacts live under `data/dicom/` and are ignored by Git except documentation.

Milestone 5 generated preprocessed NumPy volumes, metadata, and reports live under
`data/processed/` and are ignored by Git. These artefacts are for engineering validation only and
must be regenerated from synthetic or publicly available de-identified inputs.

Milestone 6 generated registration fixtures and outputs also live under `data/processed/` and are
ignored by Git. They include transforms, registered NumPy arrays, metrics, reports, and review arrays
for engineering validation only.
