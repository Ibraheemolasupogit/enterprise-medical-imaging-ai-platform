# Data Protection Design

Only synthetic or publicly available de-identified data may be used.

Future data workflows must address:

- Direct DICOM identifiers.
- Private tags.
- Burned-in identifying information.
- Dataset provenance.
- Raw, interim, and derived data separation.
- Audit logging.
- Prevention of patient data entering Git.

Milestone 1 does not implement de-identification or DICOM ingestion.
