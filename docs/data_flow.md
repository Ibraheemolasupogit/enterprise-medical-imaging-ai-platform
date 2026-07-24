# Data Flow

Milestone 2 can generate synthetic NumPy arrays for engineering fixtures. It does not ingest, transform, or store real medical images.

Milestone 3 can generate synthetic DICOM fixtures and ingest local DICOM files for header discovery, metadata extraction, structural validation, and metadata de-identification. It does not connect to PACS, DICOMweb, cloud services, or NHS systems.

Milestone 4 adds technical quality-control reports under `data/dicom/quality/`. These reports are generated artefacts and remain ignored by Git.

Milestone 5 adds deterministic CT preprocessing outputs under `data/processed/preprocessing/`.
Preprocessing consumes one explicitly selected synthetic or de-identified DICOM series, records the
source quality-control report ID/status, assembles an internal `[z, y, x]` NumPy volume, records
spacing as `[z_mm, y_mm, x_mm]`, applies configured intensity and engineering crop/pad transforms,
and writes `volume.npy`, `metadata.json`, and `preprocessing_report.md`. These derived artefacts
remain ignored by Git.

Milestone 6 adds registration outputs under `data/processed/registration/` and synthetic
registration bridge fixtures under `data/processed/registration-fixtures/`. Registration consumes
validated preprocessing directories with explicit fixed and moving roles, transforms the moving
volume into fixed-volume space, and writes registered NumPy volumes, transform JSON, metrics JSON,
metadata, Markdown reports, and lightweight review arrays. These artefacts remain ignored by Git and
are not clinical images or diagnostic overlays.

Milestone 7 adds localisation fixtures under `data/processed/localisation-fixtures/` and
localisation outputs under `data/processed/localisation/`. Localisation consumes one explicit
preprocessed or registered volume, preserves upstream provenance, and writes separate left/right ROI
arrays, overlay arrays, JSON metadata, checksums, optional synthetic-label metrics, and a Markdown
report. These outputs are ignored by Git and are atlas-baseline engineering artefacts only.

Planned - not yet implemented data zones:

- Raw external source area outside Git.
- Interim processing area outside Git.
- Dataset manifests for synthetic or publicly available de-identified data.
- Audit records for de-identification and provenance.
- Future NIfTI, deformable registration, model, and clinical review artefact zones outside Git.
- Future learned localisation, segmentation, classification, calibration, and review outputs outside
  Git.

No credentials, patient information, DICOM studies, NIfTI volumes, model weights, or restricted labels should be committed.

Generated DICOM fixtures, de-identified outputs, audit artefacts under `data/dicom/`, preprocessed
volumes, registration outputs, and localisation outputs under `data/processed/` are ignored by Git.
