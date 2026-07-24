# Data Flow

Milestone 1 does not ingest, generate, transform, or store medical images.

Planned - not yet implemented data zones:

- Raw external source area outside Git.
- Interim processing area outside Git.
- Processed derived data area outside Git.
- Dataset manifests for synthetic or publicly available de-identified data.
- Audit records for de-identification and provenance.

No credentials, patient information, DICOM studies, NIfTI volumes, model weights, or restricted labels should be committed.
