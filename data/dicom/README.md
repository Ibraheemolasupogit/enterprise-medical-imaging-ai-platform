# DICOM Working Data

Generated DICOM fixtures, de-identified outputs, and audit files under this directory are ignored by Git.

Use:

```bash
make generate-dicom-fixtures
make validate-dicom-fixtures
make verify-dicom-ingestion
make verify-dicom-quality
make verify-preprocessing
```

Only synthetic engineering fixtures or publicly available de-identified data may be used. Do not commit real patient data, restricted data, DICOM outputs, UID mapping artefacts, or audit files that might expose sensitive information.

Quality reports under `data/dicom/quality/` are generated and ignored by Git. They are technical engineering reports, not clinical image-quality assessments.

Preprocessing consumes one selected DICOM series after quality control and writes derived NumPy
artefacts under ignored `data/processed/` paths.
