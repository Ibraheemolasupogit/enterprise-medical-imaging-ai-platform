# Medical Image Preprocessing Baseline

Milestone 19 adds a deterministic classical image-content preprocessing and quality-control baseline
for already-generated preprocessed NumPy volumes. It is an engineering evidence layer only. It is not
a diagnostic image-quality assessment, reconstruction pipeline, medical-device preprocessing path, or
clinical decision-support feature.

## Relationship To Canonical Preprocessing

Canonical DICOM preprocessing remains Milestone 5. It discovers and validates DICOM series, applies
quality gates, orders slices, applies rescale slope and intercept, records geometry provenance, applies
configured clipping and normalisation, performs deterministic crop and padding, and writes
`volume.npy`, `metadata.json`, and `preprocessing_report.md`.

The M19 baseline starts after that point. It accepts one or more validated preprocessing output
directories, loads the existing `volume.npy`, computes before metrics, applies optional classical
transforms to a copy, computes after metrics, and writes separate evidence. It never overwrites or
mutates the canonical preprocessing output.

## Metrics

The baseline computes deterministic scalar intensity metrics for 2D slices and 3D volumes:

- Minimum, maximum, mean, standard deviation, and configured percentiles.
- Sharpness as Laplacian variance.
- Noise estimate as the standard deviation of a high-frequency residual from an edge-padded local
  mean filter.
- Edge density from gradient magnitude above a configured percentile threshold.
- Histogram entropy.
- Mean local contrast from the absolute high-frequency residual.

Inputs must be finite numeric NumPy arrays. No patient identifiers are read or emitted.

## Classical Transforms

The transform chain is configuration driven and recorded in evidence. The default chain starts with a
no-op source copy, then may apply:

- Mean or median denoising with an odd kernel size.
- Percentile contrast stretching.
- Global histogram equalisation.
- CLAHE-style per-axial-slice tile equalisation implemented for scalar intensity data.

Transforms are deterministic and operate on intensity arrays, not RGB or HSV image data. Output arrays
are range-managed and written as separate baseline evidence copies.

## Evidence

Each run writes:

- `preprocessing_baseline_metrics.json`
- `before_after_summary.json`
- `preprocessing_baseline_report.md`
- `evidence/checksum_manifest.json`
- transformed baseline volume copies
- bounded representative before/after PNG previews when enabled

Evidence includes source preprocessing run IDs, source metadata and volume checksum references,
policy version, transform chain, before and after metrics, metric deltas, warnings, limitations, and
a descriptive MONAI comparison.

## MONAI Comparison

The report records current segmentation and classification MONAI transforms only as a
`model-time augmentation baseline`. These transforms are training-time augmentation used by model
workflows. They are not equivalent to M19 classical preprocessing, and M19 does not invoke model
training.

## CLI

Run the baseline against one or more existing preprocessing output directories:

```bash
python3 -m medical_imaging_platform run-preprocessing-baseline \
  data/processed/preprocessing/preprocess-example \
  --config config/preprocessing_baseline.yaml \
  --output-dir reports/preprocessing_baseline \
  --json
```

The command validates each source preprocessing output before loading arrays. Overwrite protection is
enabled unless `--overwrite` is supplied.
