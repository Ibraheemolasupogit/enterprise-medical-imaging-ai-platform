# Security

Milestone 1 security controls focus on repository hygiene:

- No real patient data.
- No credentials.
- No cloud account details.
- No generated DICOM or NIfTI files.
- No model artefacts.
- Explicit research-only use boundaries.
- Basic static security scanning with Bandit.
- DICOM working data ignored under `data/dicom/`.
- Private DICOM tags removed by default during de-identification.
- Direct identifier values excluded from normal metadata output and audits.
- In-place source overwrite refused by default.

Future milestones will add controls for de-identification audit, object storage, access control, container scanning, infrastructure validation, and operational monitoring.

Metadata de-identification cannot guarantee removal of burned-in pixel identifiers. Pixel review, OCR, and pixel redaction are not implemented.
