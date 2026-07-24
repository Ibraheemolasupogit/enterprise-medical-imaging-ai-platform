# Data Flow

Milestone 2 can generate synthetic NumPy arrays for engineering fixtures. It does not ingest, transform, or store real medical images.

Milestone 3 can generate synthetic DICOM fixtures and ingest local DICOM files for header discovery, metadata extraction, structural validation, and metadata de-identification. It does not connect to PACS, DICOMweb, cloud services, or NHS systems.

Milestone 4 adds technical quality-control reports under `data/dicom/quality/`. These reports are generated artefacts and remain ignored by Git.

Planned - not yet implemented data zones:

- Raw external source area outside Git.
- Interim processing area outside Git.
- Processed derived data area outside Git.
- Dataset manifests for synthetic or publicly available de-identified data.
- Audit records for de-identification and provenance.

No credentials, patient information, DICOM studies, NIfTI volumes, model weights, or restricted labels should be committed.

Generated DICOM fixtures, de-identified outputs, and audit artefacts under `data/dicom/` are ignored by Git.
