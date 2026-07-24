# Localisation Design

This platform is a research and engineering demonstrator. Outputs are intended for technical evaluation and human review only and must not be used for clinical diagnosis or patient-management decisions.

Milestone 7 implements baseline adrenal-region localisation as an engineering placeholder. It does
not segment adrenal glands, detect lesions, classify lesions, or make anatomical adequacy claims.

## Scope

The implemented scope is deterministic left/right adrenal-region placeholder localisation for one
validated preprocessed or registered volume at a time. Inputs use the repository internal
`[z, y, x]` NumPy convention with spacing recorded as `[z_mm, y_mm, x_mm]`.

The baseline uses configured relative centres and optional physical offsets from
`config/localisation.yaml`. It extracts separate left and right ROIs and never combines sides.

## Synthetic Fixtures

`generate-localisation-fixtures` creates a preprocessing-compatible synthetic volume with binary
`left_adrenal_mask.npy` and `right_adrenal_mask.npy` engineering labels. These masks are deliberately
simple placeholders for deterministic validation and are not clinical annotations.

Fixture metadata records the centres, translation, seed, and `labels_are_clinical=false`.

## Quality Gates

Milestone 7 evaluates deterministic rule IDs:

- `LOC-QC-INP-001`: input structure and upstream status.
- `LOC-QC-GEO-001`: geometry, spacing, and axis convention.
- `LOC-QC-LR-001`: left/right centre separation.
- `LOC-QC-BND-001`: bounding-box IoU against optional synthetic masks.
- `LOC-QC-ROI-001`: empty ROI protection.
- `LOC-QC-CEN-001`: centre distance against optional synthetic masks.
- `LOC-QC-COV-001`: target coverage against optional synthetic masks.
- `LOC-QC-SWP-001`: left/right swap detection.
- `LOC-QC-PAD-001`: boundary padding fraction.
- `LOC-QC-OVR-001`: upstream quality override propagation.

Invalid geometry is rejected. Swaps, poor coverage, poor IoU, and excessive centre distance fail the
run. Missing ground truth is reported as `NOT_EVALUATED` rather than silently scored.

## Outputs

Outputs are written under ignored `data/processed/localisation/<run_id>/` by default:

- `left_roi.npy`
- `right_roi.npy`
- `localisation.json`
- `localisation_report.md`
- `left_overlay.npy`
- `right_overlay.npy`

The overlay arrays are deterministic NumPy mid-slice review artefacts with predicted centre and
bounding-box marks. They are not diagnostic visualisations.

## Limitations

The atlas baseline does not learn image features, infer patient anatomy, handle unusual protocols, or
prove localisation accuracy on real CT. It exists to establish interfaces, provenance, validation,
quality gates, and deterministic reports before PyTorch/MONAI modelling milestones.

Milestone 8 segmentation may eventually consume localisation-derived ROIs. Localisation failures,
left/right reversals, or excessive padding must be treated as upstream blockers because they can
propagate directly into segmentation samples.

Milestone 10 longitudinal analysis records localisation run IDs and statuses when available.
Left/right inconsistency or localisation failure must remain visible and can contribute to
indeterminate engineering labels.
