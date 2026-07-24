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
- No NIfTI export, general preprocessing resampling, full anatomical reorientation, deformable registration, learned localisation beyond the baseline, advanced segmentation, or classification exists.
- No API, dashboard, MLflow, Docker, Kubernetes, Terraform, or AWS implementation exists.
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
- Missing slices are inferred only when spacing evidence is reliable; metadata may be incomplete or inaccurate.

The platform is a portfolio and research demonstrator, not an approved medical device.
