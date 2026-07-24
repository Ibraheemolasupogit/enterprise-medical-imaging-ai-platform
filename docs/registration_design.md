# Registration Design

Milestone 6 implements a reproducible registration baseline for synthetic and safely de-identified
preprocessed CT-like volumes. It is technical alignment for research engineering evaluation only and
is not a diagnostic system or medical-device registration workflow.

## Fixed And Moving Roles

The fixed volume is the reference volume. The moving volume is transformed into fixed-volume space.
The pipeline requires explicit fixed and moving preprocessing directories through CLI arguments or
function parameters. It never infers previous/current order from filenames and does not silently
reverse transform direction.

Registration metadata records fixed and moving preprocessing run IDs, study and series UIDs, optional
temporal labels, preprocessing quality status, override propagation, and the direction
`moving_volume_transformed_into_fixed_volume_space`.

## Preconditions

Registration consumes Milestone 5 preprocessing outputs and validates them before SimpleITK
optimisation. Inputs must exist, validate, contain finite 3D numeric arrays, use `[z, y, x]`, have
positive spacing, include geometry metadata, not be the same preprocessing run, and not carry rejected
source quality status. Constant volumes are blocked unless explicitly allowed for controlled tests.

## NumPy And SimpleITK Geometry

Repository arrays use `[z, y, x]`. SimpleITK images use `[x, y, z]` spacing and physical coordinate
conventions. Conversion maps spacing from `[z_mm, y_mm, x_mm]` to `[x_mm, y_mm, z_mm]`, preserves
origin from source geometry when available, and preserves direction cosines when present. This does
not claim full anatomical canonicalisation or preprocessing resampling.

## Registration Modes

Supported modes are:

- `centre_of_mass`: deterministic foreground centre-of-mass translation baseline.
- `rigid`: SimpleITK Euler 3D rigid registration initialised from centre of mass.
- `rigid_then_affine`: rigid registration followed by an affine stage.

The affine stage is optional and quality gated. Affine completion is not automatically considered
success because affine transforms may distort anatomy.

## Metrics And Quality Gates

Reports distinguish intensity similarity, mask/foreground overlap, geometry plausibility, and
optimiser status. Metrics include MSE, normalised cross-correlation, mutual information, foreground
overlap, Dice, centre-of-mass distance, transform magnitude, and processing duration.

Quality rules use stable IDs such as `REG-QC-INP-001`, `REG-QC-GEO-001`, `REG-QC-CONV-001`,
`REG-QC-TRN-001`, `REG-QC-ROT-001`, `REG-QC-AFF-001`, `REG-QC-MET-001`, and `REG-QC-PAD-001`.
Statuses are `PASS`, `PASS_WITH_WARNINGS`, `FAIL`, and `REJECTED`.

Metric improvement does not prove anatomical correctness. Synthetic translation recovery does not
prove real-world performance.

## Outputs

Outputs are ignored generated artefacts under `data/processed/registration/<registration_run_id>/`:

- `registered_moving_volume.npy`
- `transform.json`
- `metrics.json`
- `registration_metadata.json`
- `registration_report.md`
- `fixed_mid_axial.npy`
- `moving_mid_axial.npy`
- `registered_mid_axial.npy`
- `overlay_mid_axial.npy`
- `difference_mid_axial.npy`

Visual review artefacts are lightweight NumPy arrays only. They are not clinical viewer screenshots.

## Explicit Non-Scope

Milestone 6 registration outputs may be consumed by Milestone 7 localisation, but registration
success remains a technical precondition rather than proof of anatomical alignment. Downstream
localisation preserves registration status, geometry, and override provenance.

Milestone 10 longitudinal analysis can consume registration run IDs and registration quality
statuses as upstream evidence. Failed registration can force `indeterminate` longitudinal labels,
and registration success still does not prove clinical anatomical correctness.

Milestone 6 does not implement deformable registration, deep-learning registration, PyTorch, MONAI,
lesion segmentation, classification, PACS/DICOMweb, notebooks, MLflow, API, dashboard, Docker,
Kubernetes, Terraform, or AWS.
