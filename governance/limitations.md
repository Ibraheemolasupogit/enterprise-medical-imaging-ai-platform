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

Milestone 10 longitudinal analysis uses synthetic masks, deterministic component matching, and
spacing-aware engineering measurements only. Synthetic change labels cannot support progression,
treatment-response, RECIST, diagnostic, triage, or patient-management claims. Measurement and
matching outputs require human engineering review and upstream quality review.

Milestone 11 API routes provide local governed research access to existing synthetic segmentation,
classification, longitudinal, and review evidence workflows. API responses are bounded engineering
summaries only. Readiness checks, checksums, successful requests, and sanitized errors do not imply
clinical validation, cybersecurity approval, deployment approval, or medical-device compliance.

Milestone 12 reviewer UI routes display local governed API outputs and collect human engineering
review decisions. UI decisions, exports, and reports are not clinical approvals, audit records for a
regulated system, diagnostic findings, RECIST assessments, or treatment-response assessments.

Milestone 14 model registry, monitoring, drift, alert, and audit evidence is local synthetic
engineering evidence only. Registry approval is human-governed and does not deploy a model. Drift
alerts trigger investigation and change-control review only; they do not prove clinical performance
deterioration. Audit JSONL evidence is not a regulated clinical audit store.
## Container And Release Limitations

Local images, SBOMs, scans, smoke tests and release manifests are engineering evidence only. They do
not establish production security, clinical safety, high availability, disaster recovery, cloud
certification, or Kubernetes readiness.
