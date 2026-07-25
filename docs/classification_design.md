# Classification Design

This platform is a research and engineering demonstrator. Outputs are intended for technical evaluation and human review only and must not be used for clinical diagnosis or patient-management decisions.

Milestone 9 implements a binary synthetic lesion-presence classification baseline for ROI-like
engineering crops. It predicts whether a repository-generated synthetic adrenal-side crop contains
the synthetic lesion mask signal. It is not benign-versus-malignant classification, not clinical
lesion detection, and not diagnostic decision support.

## Dataset Preparation

`prepare-classification-data` derives deterministic `[D, H, W]` ROI-like crops from the existing
synthetic CT fixture manifest. Each sample records the image path, binary label, label name, case,
research subject, split, scenario, side, timepoint, spacing, source checksums, and optional
localisation and segmentation provenance fields.

Training labels are:

- `0`: `no_visible_synthetic_lesion`.
- `1`: `synthetic_lesion_present`.

`indeterminate` is reserved for inference abstention only and is not a training class.

Subject-level split isolation is mandatory. If inherited synthetic splits are not binary for this
derived classification task, the preparation step creates a deterministic subject-isolated
classification split. Preparation rejects empty or single-class train, validation, or test splits.

## Model

The baseline classifier is a small original 3D convolutional network implemented in PyTorch. It
uses one image channel, configurable convolution blocks, global average pooling, dropout, and a
single fully connected logit. No pretrained weights, external model downloads, or public datasets
are used.

## Training

Training is CPU-compatible by default and records:

- random seed and device;
- dataset manifest checksum;
- configuration checksum;
- dependency versions;
- training and validation loss;
- validation AUROC, AUPRC, and recall;
- best and last state-dict checkpoints;
- model parameter count and architecture summary.

Loss is binary cross entropy with logits and optional positive-class weighting. Early stopping uses
validation AUPRC. Validation and test splits must remain subject-isolated.

## Calibration And Thresholds

Calibration is fit on validation logits only. Supported methods are `none`, Platt scaling, and
isotonic regression with automatic fallback when validation data are insufficient. Threshold
selection is also validation-only and supports fixed, Youden, minimum sensitivity, minimum NPV, and
maximum false-positive policies.

Inference applies calibration, thresholding, and an abstention interval. Probabilities inside the
configured abstention interval return the inference-only label `indeterminate`.

Milestone 10 longitudinal analysis can propagate classification run IDs, calibration evidence, and
abstention status. Abstained classification must not be converted into a confident longitudinal
engineering label.

## Outputs

Generated artefacts are ignored by Git:

- `ml/datasets/classification/<dataset_id>/`
- `ml/experiments/classification/<experiment_id>/`
- `ml/experiments/classification-inference/<run>/`

Experiment outputs include best and last checkpoints, calibration evidence, threshold policy,
configuration, training history, evaluation metrics, per-sample predictions, a model manifest, and
`classification_report.md`.

## Quality Gates

Milestone 9 quality rules use stable IDs:

- `CLS-QC-DATA-001`: dataset manifest validity.
- `CLS-QC-SPLIT-001`: subject-level leakage.
- `CLS-QC-TRAIN-001`: finite training completion.
- `CLS-QC-AUC-001`: validation AUROC threshold.
- `CLS-QC-PRC-001`: validation AUPRC threshold.
- `CLS-QC-REC-001`: validation recall threshold.
- `CLS-QC-BRI-001`: validation Brier-score ceiling.
- `CLS-QC-FN-001`: test false-negative ceiling.
- `CLS-QC-CAL-001`: calibration evidence presence.
- `CLS-QC-THR-001`: validation threshold evidence presence.
- `CLS-QC-ABS-001`: abstention configuration validity.
- `CLS-QC-CHK-001`: checkpoint and evidence completeness.

These gates are engineering controls for deterministic synthetic fixtures only. They are not
clinical acceptance criteria.

## Limitations

The baseline learns from simplified synthetic intensities, masks, and crops. Synthetic AUROC,
AUPRC, recall, Brier score, calibration diagnostics, and threshold behavior do not demonstrate
clinical performance. ROI selection, localisation errors, segmentation errors, class imbalance,
miscalibration, threshold choices, and abstention policy can all affect downstream research
workflows and require human engineering review.

Milestone 11 exposes classification inference through `POST /v1/classification/predict` for local
research use only. The API requires checkpoint, calibration, and threshold-policy artefacts, returns
probabilities, checksums, engineering labels, and abstention state, and does not return weights or
training data. The endpoint is not benign-versus-malignant classification and is not diagnostic.

Milestone 12 displays classification API summaries in the reviewer UI. The `indeterminate` label and
abstention reason remain prominent and are not collapsed into a binary result.
