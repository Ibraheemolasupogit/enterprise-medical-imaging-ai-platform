# Synthetic Data Design

Milestone 2 adds small deterministic synthetic CT-like arrays for engineering tests.

The generated volumes include:

- Air-like background values.
- A simplified soft-tissue-like body ellipsoid.
- Simplified internal structures.
- Left and right adrenal-region placeholders.
- Optional synthetic lesion masks.
- Previous/current longitudinal pairs.

These arrays are not clinically realistic CT scans. They are geometry and metadata fixtures for future pipeline development, validation, and review demonstrations. They cannot support clinical-performance claims.

Supported longitudinal scenarios:

- Stable lesion.
- Increased lesion volume.
- Reduced lesion volume.
- New lesion.
- Resolved lesion.
- Translated anatomy without registration.

The generator writes NumPy `.npy` arrays and JSON metadata only. It does not generate DICOM or NIfTI files.
