# Canonical platform demonstration

M23 consolidates existing functionality. Run from the repository root with the development dependency
set installed: `make platform-demo`. To repeat, supply a new empty directory with
`PLATFORM_DEMO_DIR=/tmp/platform-demo-repeat`; existing evidence is never silently overwritten.

## Actual flow

1. Generate twelve deterministic 32³ synthetic cases with subject-level provenance.
2. Separately generate four tiny DICOM slices with fixed UIDs. Run DICOM QC inside preprocessing,
   classical preprocessing comparison and atlas localisation. Preserve localisation findings,
   including padding failures on this deliberately small fixture.
3. Prepare segmentation data from synthetic volumes/reference masks. Train the existing U-Net for
   one epoch and infer previous/current test volumes. Feed those **predicted masks** into longitudinal
   measurements with actual segmentation run references and quality status.
4. Prepare classification crops using the existing reference-based label/crop contract. Its
   deterministic split repair may differ from the source dataset splits; each dataset manifest is
   authoritative and remains subject-isolated. Train CNN/ViT using the shared benchmark workflow.
5. Generate and validate both embedding indexes. Query/evaluate exact retrieval and compare both
   encoders using the same classification membership and retrieval policy.
6. Prepare reference-derived axial boxes and train/evaluate Faster R-CNN/Tiny DETR with existing
   detection benchmarking. Detectors do not consume classifier or segmentation outputs.
7. Exercise health/readiness, embeddings search and segmentation preview through the actual API.
   Export reviewer-visible responses and a separate checksum-bound `needs_secondary_review` decision.
   This automated decision is a demonstration record, not human acceptance.
8. Write technical comparison evidence and validate the final index against generated files.

The demo's PASS means orchestration and integrity succeeded. Individual quality statuses remain
visible; it does not mean every model is accurate or every upstream QC result passed. Localisation
is not used downstream when its tiny-fixture quality checks fail. Prediction masks may be empty.

## Reproducibility and evidence

Seeds, membership, tensor shapes, ordering and model policies are fixed. CPU execution uses one torch
thread and one epoch. Paths, timestamps and measured timings can vary; bitwise equality of complete
reports across hardware is not promised. No pretrained weights or external assets are downloaded.
`evidence_index.json` carries canonical metadata integrity, every generated file hash, and additive
common-envelope references. `validate_demo(Path(...))` rechecks these references and hashes.

The normal test suite executes the real bounded workflow and tests tampering and overwrite rejection.
The installed and module command paths share `platform-demo`. Earlier `make demo` uses reference masks
for longitudinal analysis and remains available for backwards compatibility.

## Capability status

Implemented = code exists. Executed/validated = a named run/check has recorded evidence.
Simulated = deterministic engineering scenario. Static design = policy/design checked without a live
workload. Target state = future operational architecture. AWS remains target state; Helm static checks
and operations simulations do not become live deployments because the CPU model demo passed.
