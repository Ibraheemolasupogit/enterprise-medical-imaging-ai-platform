# Data Protection Design

Only synthetic or publicly available de-identified data may be used.

Milestone 3 supports locally generated synthetic DICOM fixtures and metadata-focused de-identification for synthetic or already de-identified public data. It does not download public data, connect to PACS/DICOMweb, or process real patient data.

Future data workflows must address:

- Direct DICOM identifiers.
- Private tags.
- Burned-in identifying information.
- Dataset provenance.
- Raw, interim, and derived data separation.
- Audit logging.
- Prevention of patient data entering Git.

Metadata de-identification cannot guarantee removal of burned-in pixel identifiers. Future pixel-level review or redaction is required before working with public DICOM data that may contain burned-in annotations.

Milestone 4 quality control escalates burned-in annotation metadata but does not inspect pixels for PHI.

Milestone 5 preprocessing writes derived NumPy volumes and reports under ignored processed-data
directories. Preprocessing metadata must remain technical and must not include patient names, patient
IDs, accessions, institutions, or other direct identifiers. It does not perform OCR, pixel PHI
redaction, or confidentiality-profile certification; source data must already be synthetic or
properly de-identified before use.
