# Governance Limitations

This repository is a portfolio and research demonstrator.

It is not:

- An approved medical device.
- Validated for NHS deployment.
- Suitable for diagnosis.
- Suitable for patient management.
- A source of clinical performance claims.

Only synthetic or publicly available de-identified data may be used. Malignancy-versus-benignity classification is out of scope unless a suitable labelled public dataset is identified and its limitations are documented.

Milestone 2 synthetic volumes are engineering fixtures, not clinically realistic CT scans. They cannot support clinical-performance claims.

Milestone 3 DICOM de-identification is metadata-focused and does not establish full confidentiality-profile compliance. Burned-in pixel identifiers require manual review or future pixel-redaction capability.
