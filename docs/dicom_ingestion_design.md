# DICOM Ingestion Design

Milestone 3 implements a safe DICOM ingestion foundation for synthetic fixtures and already de-identified public data.

Implemented:

- Recursive local-file discovery.
- Header-only discovery by default.
- Grouping by `StudyInstanceUID` and `SeriesInstanceUID`.
- Safe technical metadata extraction.
- Deterministic slice ordering strategy.
- Basic structural validation findings.
- Synthetic DICOM CT fixture generation for tests and local validation.
- Technical DICOM quality-control orchestration in Milestone 4.

Not implemented:

- PACS, DICOMweb, or NHS system integration.
- DICOM-to-NIfTI conversion.
- Hounsfield-unit conversion.
- Resampling, preprocessing, registration, localisation, segmentation, or classification.
- Clinical-quality DICOM validation or regulatory compliance.

Milestone 4 quality control remains an engineering check. It is not a clinical image-quality assessment.

Source data must remain outside Git. Generated fixtures and de-identified outputs under `data/dicom/` are ignored.
