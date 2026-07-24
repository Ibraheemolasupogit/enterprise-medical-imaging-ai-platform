# Data Provenance

Milestone 2 synthetic data provenance is recorded through deterministic manifests.

Each manifest record includes:

- Dataset ID and version.
- Case ID.
- Research subject ID.
- Previous and current study IDs.
- Scenario.
- Volume shape and voxel spacing.
- Lesion side and lesion volumes.
- Random seed.
- Generator version.
- File paths.
- SHA-256 checksums.

Future public datasets must add source, licence, access date, de-identification status, redistribution restrictions, annotation provenance, and dataset limitations.

Burned-in pixel identifiers are not addressed by metadata-only controls and require separate image-level review in future milestones.
