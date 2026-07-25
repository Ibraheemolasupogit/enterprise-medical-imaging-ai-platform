# Roadmap

1. Repository foundation. Complete.
2. Synthetic and public-data foundation. Complete for synthetic-data foundation and public-data selection criteria only; no public data is downloaded.
3. DICOM ingestion and governance. Complete for local synthetic fixtures, metadata extraction, structural validation, and metadata de-identification only.
4. Imaging quality control. Complete for technical DICOM engineering checks only.
5. Preprocessing. Complete for deterministic NumPy CT preprocessing foundation only.
6. Registration. Complete for SimpleITK centre-of-mass, rigid, and affine baselines only.
7. Baseline localisation. Complete for deterministic adrenal-region placeholder localisation only; no segmentation.
8. Synthetic segmentation baseline. Complete for small MONAI 3D U-Net on synthetic masks only.
9. Classification and calibration. Complete for binary synthetic lesion-presence classification only.
10. Longitudinal analysis. Complete for governed synthetic engineering labels only.
11. API foundation. Complete for governed local FastAPI service only; review dashboard remains planned.
12. Reviewer UI foundation. Complete for local Streamlit API-consuming review interface only.
13. MLOps platform.
14. Docker and Kubernetes.
15. AWS deployment blueprint.
16. Clinical AI assurance.
17. Final portfolio packaging.

Milestones 1-12 are implemented in the current repository state. Milestone 12 does not implement
deformable registration, NIfTI export, general preprocessing resampling, learned localisation beyond
the baseline, advanced segmentation, benign-versus-malignant classification, clinical lesion
detection, RECIST, treatment-response assessment, authentication, persistent dashboard workflows, cloud
deployment, or clinical decision support.
