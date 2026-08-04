# Limitations

Current limitations:

- Synthetic CT-like engineering fixtures exist, but real patient data must not be loaded.
- DICOM ingestion is limited to local-file discovery, metadata extraction, synthetic fixtures, structural validation, and metadata de-identification.
- No PACS, DICOMweb, NHS integration, or network DICOM workflow exists.
- Metadata de-identification cannot guarantee removal of burned-in pixel identifiers.
- Preprocessing is limited to deterministic NumPy assembly, technical rescale conversion, intensity transforms, engineering crop/pad operations, metadata, and validation.
- Registration is limited to SimpleITK centre-of-mass, rigid, and affine engineering baselines for preprocessed volumes.
- Localisation is limited to deterministic atlas-style left/right adrenal-region placeholders for synthetic engineering validation.
- Segmentation is limited to a small PyTorch/MONAI 3D U-Net trained on synthetic engineering lesion masks.
- Classification is limited to a small PyTorch 3D CNN trained on synthetic ROI-like crops for binary synthetic lesion presence only.
- Longitudinal analysis is limited to synthetic mask measurements, deterministic matching, and engineering change labels.
- No NIfTI export, general preprocessing resampling, full anatomical reorientation, deformable registration, learned localisation beyond the baseline, advanced segmentation, benign-versus-malignant classification, RECIST, treatment-response assessment, or clinical diagnosis exists.
- A local governed FastAPI research API and Streamlit reviewer UI exist, but no authentication, persistent audit
  log, MLflow, Docker, Kubernetes, Terraform, or AWS implementation exists.
- No clinical validation has been performed.
- Synthetic volumes are not clinically realistic CT scans.
- Synthetic lesion shapes do not represent the diversity of real pathology.
- Synthetic data cannot support clinical-performance claims.
- Milestone 3 does not establish regulatory compliance or full DICOM confidentiality-profile compliance.
- Milestone 4 quality scores are engineering indicators only and do not establish diagnostic adequacy.
- Milestone 5 preprocessing reports are engineering provenance only and do not establish diagnostic image standardisation.
- Milestone 6 registration reports are engineering provenance only. Optimiser convergence, metric improvement, and synthetic translation recovery do not establish anatomical correctness.
- Milestone 7 localisation reports are engineering provenance only. Synthetic placeholder masks and
  atlas centres do not establish adrenal localisation accuracy or anatomical correctness.
- Milestone 8 segmentation reports are engineering evidence only. Synthetic Dice, recall, volume
  error, and surface distances do not establish clinical lesion segmentation performance.
- Milestone 9 classification reports are engineering evidence only. Synthetic AUROC, AUPRC, recall,
  Brier score, calibration diagnostics, threshold behavior, and abstention rates do not establish
  clinical lesion detection or diagnostic classification performance.
- Milestone 10 longitudinal reports are engineering evidence only. Synthetic `new`, `increased`,
  `stable`, `reduced`, `resolved`, and `indeterminate` labels do not establish progression,
  response, RECIST category, diagnosis, or patient-management suitability.
- Milestone 11 API responses are engineering summaries only. API availability, readiness, checksums,
  and successful requests do not establish clinical safety, diagnostic performance, or deployment
  approval.
- Milestone 12 reviewer decisions are engineering workflow artefacts only. They are not clinical
  approval, diagnosis, treatment-response assessment, RECIST compliance, or medical-device evidence.
- Missing slices are inferred only when spacing evidence is reliable; metadata may be incomplete or inaccurate.

The platform is a portfolio and research demonstrator, not an approved medical device.
## Container Release Limitations

Milestone 13 container images and release evidence are local engineering artefacts only. They do not
represent production deployment readiness, internet-facing security, clinical operational approval,
cloud certification, high availability, disaster recovery, or Kubernetes readiness.
