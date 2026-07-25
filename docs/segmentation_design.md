# Segmentation Design

This platform is a research and engineering demonstrator. Outputs are intended for technical evaluation and human review only and must not be used for clinical diagnosis or patient-management decisions.

Milestone 8 implements the first learned model in the repository: a small MONAI 3D U-Net trained
only on synthetic engineering lesion masks. It is a synthetic lesion segmentation baseline, not a
clinical lesion segmenter.

## Dataset Preparation

`prepare-segmentation-data` converts the existing synthetic longitudinal dataset into
segmentation-ready samples. Each previous and current timepoint becomes one sample with:

- `image.npy`
- `lesion_mask.npy`
- case, subject, split, scenario, spacing, source checksum, and optional localisation provenance

Subject-level splits are reused from the Milestone 2 split file. Negative samples are retained and
must remain part of training, validation, and test evidence.

## Model

The default model is a small MONAI `UNet` with `spatial_dims=3`, one input channel, one output
channel, configurable channels and strides, and no pretrained weights. The default verification
configuration is deliberately small for CPU execution.

The model emits logits. Inference applies sigmoid, thresholding, and optional simple
post-processing. The default verification threshold is conservative to avoid accepting
all-foreground predictions from a tiny synthetic smoke-test model.

## Training And Evaluation

Training records seed, dependency versions, device, configuration checksum, dataset manifest
checksum, parameter count, training history, best epoch, validation metrics, test metrics, and
checkpoint checksums.

Evaluation reports per-case metrics, aggregate metrics, positive-case and negative-case metrics,
scenario summaries, and failure cases. Test metrics are evidence only and are not used to tune the
checkpoint.

## Quality Gates

Milestone 8 quality rules use stable IDs:

- `SEG-QC-DATA-001`: dataset manifest validity.
- `SEG-QC-SPLIT-001`: subject-level leakage.
- `SEG-QC-TRAIN-001`: training completion.
- `SEG-QC-LOSS-001`: finite losses.
- `SEG-QC-DICE-001`: validation Dice threshold.
- `SEG-QC-RECALL-001`: test recall threshold.
- `SEG-QC-FP-001`: false-positive voxel burden.
- `SEG-QC-VOL-001`: relative volume error.
- `SEG-QC-CHK-001`: checkpoint integrity.
- `SEG-QC-REP-001`: evidence completeness.

These thresholds are engineering gates for deterministic synthetic fixtures only. They are not
clinical acceptance criteria.

## Outputs

Generated artefacts are ignored by Git:

- `ml/datasets/segmentation/<dataset_id>/`
- `ml/experiments/segmentation/<experiment_id>/`
- `ml/experiments/segmentation-inference/<run>/`

Checkpoints store state dictionaries only. Architecture and configuration are recorded separately in
JSON evidence.

## Limitations

Synthetic masks are simple engineering fixtures and do not represent scanner diversity, protocol
diversity, lesion appearance diversity, radiologist annotation variability, or clinical pathology.
Synthetic Dice, recall, and surface distances do not demonstrate clinical performance. Localisation
errors can propagate into future segmentation workflows, although Milestone 8 trains directly from
synthetic lesion masks. Downstream classification workflows must treat segmentation outputs as
provenance and risk context, not as diagnostic evidence.

Milestone 10 longitudinal analysis consumes lesion masks and segmentation quality statuses as
upstream evidence. Failed segmentation can force `indeterminate` labels, and synthetic segmentation
metrics must not be converted into clinical progression or response claims.

Milestone 11 exposes segmentation inference through `POST /v1/segmentation/predict` for local
research use only. The API requires a configured checkpoint, enforces local input-root controls,
returns bounded probability and mask summaries, and can optionally persist a predicted mask under an
ignored local output directory. The endpoint must not be used as a diagnostic segmentation service.

Milestone 12 displays segmentation API summaries in the reviewer UI. The UI may show bounded input
slice visualisation and API-returned output references, but it does not display model weights or run
segmentation inference directly.
