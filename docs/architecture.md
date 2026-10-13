# Platform architecture

M1–M22 implement imaging pipelines, supervised model workflows, benchmarks, retrieval and assurance.
M23 consolidates command routing, reviewer access, artifact binding, policy enforcement and a bounded
CPU demonstration. It introduces no new model, task family, external dataset or ML framework.

The [current architecture diagram](architecture/final_end_to_end_architecture.md) shows actual data
dependencies. Synthetic DICOM flows through QC/preprocessing/localisation. Independent synthetic
reference annotations prepare segmentation, classification and detection datasets. Trained
classifiers supply embeddings; predicted segmentation masks can supply longitudinal measurements.

## Boundaries

- Domain modules own their typed configurations, datasets, validation and evaluation.
- The shared CLI registers each command once; installed and module invocation are equivalent.
- API services bound local inputs and expose summaries/provenance plus bounded segmentation previews.
- Streamlit calls the API; it does not load model checkpoints or execute inference.
- Reviewer decisions bind to an artifact version where available and remain separate from predictions.
- Additive evidence envelopes cross-reference domain artifacts without replacing their validators.
- Model lifecycle approval is manual. Monitoring and incident scenarios remain simulations.
- Containers/Helm support local assurance; Terraform describes AWS target state without deployment.

## Remaining limitations

No general preprocessing resampling/anatomical canonicalisation, NIfTI export, deformable
registration, learned adrenal localisation, production authentication/audit store, live cloud
execution, clinical validation, automatic model promotion or automatic retraining is implied.
See [limitations](limitations.md), [evidence contracts](evidence_contracts.md),
[canonical demo](platform_demo.md) and [local startup](local_startup.md).
