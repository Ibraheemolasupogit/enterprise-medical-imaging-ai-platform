# Data Protection Design

Only synthetic or publicly available de-identified data may be used.

Milestone 2 uses locally generated synthetic arrays only. It does not download public data or process real DICOM/NIfTI studies.

Future data workflows must address:

- Direct DICOM identifiers.
- Private tags.
- Burned-in identifying information.
- Dataset provenance.
- Raw, interim, and derived data separation.
- Audit logging.
- Prevention of patient data entering Git.

Milestone 2 does not implement de-identification or DICOM ingestion.
