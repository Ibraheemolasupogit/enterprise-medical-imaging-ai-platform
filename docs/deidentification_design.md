# De-Identification Design

Milestone 3 implements metadata-focused de-identification for research fixtures and already de-identified public datasets.

Default policy:

- Remove direct identifier tags where possible.
- Replace `PatientName` and `PatientID` with generated research identifiers.
- Remap study, series, and SOP UIDs.
- Remove all private tags by default.
- Record audit data without removed identifier values.

UID remapping is deterministic within one de-identification run and preserves study/series/SOP relationships. The mapping does not reveal the source UID and is not emitted in normal CLI logs.

Limitations:

- This does not claim full DICOM confidentiality-profile compliance.
- Metadata de-identification cannot guarantee removal of burned-in pixel identifiers.
- `BurnedInAnnotation=YES` is flagged for manual review.
- OCR and pixel redaction are not implemented.
- Public datasets still require licence, provenance, and de-identification review.
