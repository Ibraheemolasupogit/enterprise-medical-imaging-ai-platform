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

Milestone 8 adds synthetic segmentation datasets under `ml/datasets/segmentation/`, segmentation
experiments under `ml/experiments/segmentation/`, and inference smoke-test outputs under
`ml/experiments/segmentation-inference/`. Segmentation consumes synthetic volumes and lesion masks,
preserves subject-level split metadata and checksums, trains a CPU-compatible MONAI U-Net, and writes
state-dict checkpoints plus structured experiment evidence. These artefacts are ignored by Git and
are not clinical model outputs.

Milestone 9 adds synthetic classification datasets under `ml/datasets/classification/`,
classification experiments under `ml/experiments/classification/`, and inference smoke-test outputs
under `ml/experiments/classification-inference/`. Classification consumes synthetic adrenal-side
ROI-like crops and synthetic lesion-presence labels, preserves subject-level split metadata and
checksums, trains a CPU-compatible PyTorch 3D CNN, fits calibration and threshold policy on
validation data only, and writes state-dict checkpoints plus structured experiment evidence. These
artefacts are ignored by Git and are not clinical model outputs.

Planned - not yet implemented data zones:

- Raw external source area outside Git.
- Interim processing area outside Git.
- Dataset manifests for synthetic or publicly available de-identified data.
- Audit records for de-identification and provenance.
- Future NIfTI, deformable registration, model, and clinical review artefact zones outside Git.
- Future learned localisation, advanced segmentation, clinical classification, calibration, and review outputs outside
  Git.

No credentials, patient information, DICOM studies, NIfTI volumes, model weights, or restricted labels should be committed.

Generated DICOM fixtures, de-identified outputs, audit artefacts under `data/dicom/`, preprocessed
volumes, registration outputs, and localisation outputs under `data/processed/` are ignored by Git.
Generated segmentation datasets, checkpoints, and inference artefacts under `ml/datasets/` and
`ml/experiments/` are ignored by Git. Generated classification datasets, checkpoints, calibration
artefacts, threshold policies, and inference artefacts under the same `ml/` areas are also ignored
by Git.

Milestone 10 adds longitudinal analysis outputs under `ml/experiments/longitudinal/`. Longitudinal
analysis consumes explicit previous/current synthetic lesion masks and spacing, records pair
metadata, propagates upstream registration/localisation/segmentation/classification status, writes
measurements, matches, change labels, quality findings, summaries, Markdown reports, and
deterministic NumPy review arrays. These artefacts are ignored by Git and are not clinical
progression, response, or RECIST outputs.

Milestone 11 adds local API outputs under `ml/experiments/api/` when `persist_output` is enabled.
The API consumes explicit `.npy` inputs from configured local roots or bounded JSON arrays, invokes
existing segmentation, classification, or longitudinal code paths, and returns bounded JSON
summaries. Read-only review endpoints validate evidence checksums before returning public summaries.
The API does not expose absolute local paths, raw images, masks, probability maps, model weights,
credentials, DICOM files, PACS data, or cloud resources.

Milestone 12 adds reviewer-session exports under `reports/generated/reviewer-sessions/`. The
reviewer UI consumes governed API JSON responses and bounded local uploads, keeps model output
separate from reviewer decisions, and writes stable review-decision JSON, bounded evidence summaries,
and Markdown reports. These exports are ignored by Git and are not clinical approval records.
## Container Data Flow

Milestone 13 packages the existing API and reviewer UI data flow into local containers. Generated
datasets, experiments, checkpoints and reviewer exports are excluded from image build contexts;
evidence and checkpoints are mounted read-only when needed, while outputs are written only to
explicit generated directories.
