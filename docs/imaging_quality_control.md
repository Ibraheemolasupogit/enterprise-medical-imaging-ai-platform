# Imaging Quality Control

Milestone 4 implements technical DICOM quality-control checks for synthetic and safely de-identified CT DICOM series.

Implemented checks include:

- Accepted modality metadata.
- Expected body-region metadata allowlist.
- Study, series, and SOP UID consistency.
- Rows, columns, pixel spacing, slice thickness, and orientation consistency.
- Deterministic slice ordering and slice-gap analysis.
- Duplicate slice positions and duplicate instance numbers.
- Pixel data presence and pixel-array readability when full pixel validation is requested.
- Pixel dimensions, constant slices, finite values, and configured engineering value bounds.
- Transfer syntax reporting.
- Burned-in annotation escalation.
- Private-tag presence reporting.

The quality score is an engineering data-quality indicator only. A passing score does not mean a scan is diagnostically adequate.

Milestone 5 preprocessing consumes this quality status before volume assembly. `REJECTED` series and
critical findings stop preprocessing. Blocking `FAIL` status requires an explicit engineering
override and is recorded in preprocessing metadata.

Limitations:

- No clinical image-quality assessment.
- No Hounsfield-unit conversion.
- No DICOM-to-NIfTI conversion.
- No OCR or pixel redaction.
- No radiologist validation.
- No regulatory, NHS, or medical-device compliance.
