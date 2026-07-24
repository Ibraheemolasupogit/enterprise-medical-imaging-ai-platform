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

Milestone 4 quality control is not clinical image-quality assessment. Passing technical checks does not mean a scan is diagnostically adequate.

Milestone 5 preprocessing is deterministic engineering preprocessing only. It assembles one selected
DICOM series into a NumPy volume, records geometry, applies technical intensity transforms, and can
crop or pad by configured non-anatomical rules. It does not perform diagnostic CT standardisation,
NIfTI conversion, spatial resampling, anatomical reorientation, registration, adrenal localisation,
segmentation, classification, or clinical inference.

Milestone 6 registration is technical alignment for research engineering evaluation only. It uses
centre-of-mass, rigid, and affine baselines on preprocessed volumes. Optimiser convergence, improved
similarity metrics, and synthetic translation recovery do not prove anatomical correctness. No
deformable, anatomy-constrained, radiologist-validated, diagnostic, or medical-device registration
is implemented.

Milestone 7 localisation is a deterministic atlas/geometry baseline for configured left/right
adrenal-region placeholders. It uses synthetic engineering masks only when optional ground truth is
provided. It does not implement learned localisation, organ segmentation, lesion detection,
classification, anatomical correctness validation, or diagnostic visualisation.

Milestone 8 segmentation is a small PyTorch/MONAI baseline trained only on synthetic engineering
lesion masks. Synthetic Dice, recall, volume error, and surface metrics cannot support clinical
claims. No public-data training, radiologist validation, lesion classification, diagnosis, or
medical-device compliance is implemented.

Milestone 9 classification is a small PyTorch baseline trained only on synthetic ROI-like crops for
binary synthetic lesion presence. Synthetic AUROC, AUPRC, recall, calibration diagnostics,
threshold behavior, and abstention outputs cannot support clinical claims. No real-patient
classification, benign-versus-malignant classification, radiologist validation, diagnosis,
triage, or medical-device compliance is implemented.
