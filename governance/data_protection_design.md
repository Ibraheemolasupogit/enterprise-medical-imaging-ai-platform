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
