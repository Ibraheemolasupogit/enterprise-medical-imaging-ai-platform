# Limitations

Current Milestone 1 limitations:

- Synthetic CT-like engineering fixtures exist, but no real medical images are loaded or processed.
- DICOM ingestion is limited to local-file discovery, metadata extraction, synthetic fixtures, structural validation, and metadata de-identification.
- No PACS, DICOMweb, NHS integration, or network DICOM workflow exists.
- Metadata de-identification cannot guarantee removal of burned-in pixel identifiers.
- No preprocessing, registration, localisation, segmentation, or classification exists.
- No API, dashboard, MLflow, Docker, Kubernetes, Terraform, or AWS implementation exists.
- No clinical validation has been performed.
- Synthetic volumes are not clinically realistic CT scans.
- Synthetic lesion shapes do not represent the diversity of real pathology.
- Synthetic data cannot support clinical-performance claims.
- Milestone 3 does not establish regulatory compliance or full DICOM confidentiality-profile compliance.

The platform is a portfolio and research demonstrator, not an approved medical device.
