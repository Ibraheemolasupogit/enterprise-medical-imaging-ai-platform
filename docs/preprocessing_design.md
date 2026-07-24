# CT Preprocessing Design

Milestone 5 implements a deterministic preprocessing foundation for synthetic or publicly
available de-identified CT DICOM series. It is an engineering demonstrator, not a diagnostic
system or medical-device preprocessing pipeline.

## Scope

The pipeline consumes one selected DICOM series, runs the existing quality-control gate, assembles a
3D NumPy volume in internal `[z, y, x]` order, applies technical rescale slope/intercept conversion,
applies configurable intensity clipping and normalisation, applies deterministic engineering crop and
padding operations, and writes ignored processed artefacts under `data/processed/`.

The pipeline does not perform PACS/DICOMweb access, OCR or pixel redaction, NIfTI export,
registration, anatomical reorientation, resampling, adrenal localisation, segmentation,
classification, model training, or clinical inference.

## Quality Gate

Preprocessing runs the Milestone 4 DICOM quality checks before assembly. `REJECTED` series and
critical failed findings stop the run with CLI exit code `3`. Blocking `FAIL` status stops with exit
code `2` unless an explicit engineering override is supplied and the preprocessing policy allows
overrides. Override use and reason are recorded in metadata.

## Volume Assembly

The selected series is validated for a single study UID and series UID, consistent rows/columns,
compatible pixel representation, transfer syntax, and present `PixelData`. Slice ordering reuses the
ingestion ordering policy and records SOP Instance UID to output slice index mappings. Ambiguous
ordering stops by default.

Pixel decoding happens after ordering and validation. Per-slice `RescaleSlope` and
`RescaleIntercept` are applied to a floating-point array. Missing slope/intercept values use
configured defaults and record affected SOP UIDs. The output terminology is `CT-like rescaled
intensity` unless future dataset evidence justifies stricter HU language.

## Geometry

Internal arrays use `[z, y, x]`; spacing metadata is `[z_mm, y_mm, x_mm]`. In-plane spacing comes from
`PixelSpacing`. Slice spacing is estimated from Image Position Patient projected along the slice
normal when reliable, with `SliceThickness` fallback when needed. Orientation cosines, slice normal,
source positions, spacing source, min/median/max gap, irregularity, and fallback flags are preserved
for later registration and NIfTI policy work.

Milestone 5 does not anatomically reorient volumes and does not resample.

## Intensity And Crop Policy

Configured profiles include `none`, `abdominal_soft_tissue`, `wide_ct`, and
`adrenal_engineering`. Normalisation modes are `none`, `minmax`, and `zscore`, with deterministic
zero-output fallbacks for constant volumes.

Crop modes are `none`, `non_background`, `centre`, and `fixed`. These are engineering operations only,
not organ localisation. Crop bounds, offsets, pad widths, padding value, and final shape are recorded.

## Outputs

Each run writes:

- `volume.npy`
- `metadata.json`
- `preprocessing_report.md`

The metadata records run ID, study/series UID, source quality report ID/status, shape, spacing,
axis order, geometry, pixel conversion, intensity transform, crop/padding trace, slice counts, SOP
UID-to-index mapping, output paths, checksums, policy version, timestamp, warnings, and override use.

Generated artefacts are ignored by Git and must not contain direct identifiers.
