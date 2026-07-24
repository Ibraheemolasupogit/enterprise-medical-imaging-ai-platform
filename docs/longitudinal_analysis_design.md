# Longitudinal Analysis Design

This platform is a research and engineering demonstrator. Outputs are intended for technical evaluation and human review only and must not be used for clinical diagnosis or patient-management decisions.

Milestone 10 implements governed longitudinal lesion analysis for paired previous/current synthetic
CT engineering masks. It measures lesion masks, matches previous and current components, calculates
physical changes, assigns synthetic engineering labels, propagates upstream quality, and writes
deterministic evidence. It does not implement RECIST, disease progression, treatment response,
cancer diagnosis, radiologist equivalence, or medical-device readiness.

## Inputs

`analyse-longitudinal-pair` accepts explicit previous/current binary 3D masks, spacing in
`[z_mm, y_mm, x_mm]`, case and research-subject metadata, anatomical side, temporal labels, optional
upstream run IDs, and upstream quality statuses. Inputs must be subject-consistent, side-consistent,
temporally ordered, binary, finite, spacing-aware, and free of direct identifiers.

Geometry must match or provide an explicit registration run ID. Registration, localisation,
segmentation, classification, calibration, and abstention statuses remain visible in final evidence.

## Measurements

Measurements are deterministic engineering measurements:

- voxel count;
- physical volume in mm3 and mL;
- bounding-box dimensions in mm;
- maximum 3D Euclidean diameter;
- axial maximum diameter;
- voxel and physical centroids;
- connected-component count;
- empty-mask status.

These are not validated clinical measurements.

## Matching

The baseline extracts connected components and computes centroid distance, bounding-box IoU, mask
Dice, and mask IoU. It uses configurable centroid and overlap weights, then applies deterministic
greedy one-to-one assignment with stable tie-breaking. Ambiguous equal-score candidates are marked
and force indeterminate change labels.

## Change Labels

Supported engineering labels are:

- `new`;
- `increased`;
- `stable`;
- `reduced`;
- `resolved`;
- `indeterminate`.

Matched lesions report absolute and percentage volume change, absolute and percentage diameter
change, centroid displacement, overlap Dice, and overlap IoU. Thresholds are configurable in
`config/longitudinal.yaml`. Small denominators produce explicit `null` percentage changes rather
than inflated conclusions.

## Quality Gates

Milestone 10 quality rules use stable IDs:

- `LNG-QC-PAIR-001`;
- `LNG-QC-TIME-001`;
- `LNG-QC-SIDE-001`;
- `LNG-QC-GEO-001`;
- `LNG-QC-REG-001`;
- `LNG-QC-SEG-001`;
- `LNG-QC-MATCH-001`;
- `LNG-QC-MEAS-001`;
- `LNG-QC-CHANGE-001`;
- `LNG-QC-LABEL-001`;
- `LNG-QC-PROV-001`;
- `LNG-QC-CHK-001`.

Failed registration, failed segmentation, incompatible geometry, ambiguous matching, and
classification abstention can force indeterminate outputs. These gates are engineering controls
only.

## Outputs

Generated outputs are ignored by Git under `ml/experiments/longitudinal/<analysis_id>/`:

- `pair_manifest.json`;
- `lesion_measurements.json`;
- `lesion_matches.json`;
- `longitudinal_changes.json`;
- `quality_findings.json`;
- `longitudinal_summary.json`;
- `longitudinal_report.md`;
- `analysis_manifest.json`;
- `review_arrays/*.npy`.

Review arrays are deterministic NumPy artefacts only. There is no GUI, dashboard, notebook, or
clinical viewer in this milestone.
