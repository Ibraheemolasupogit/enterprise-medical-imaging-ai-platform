# Medical Image Detection Design

Milestone 21 adds a synthetic 2D axial lesion detection baseline for research and engineering
evaluation only. It derives slice images and bounding boxes from existing repository-generated 3D
synthetic volumes and lesion masks. It does not introduce external data, clinical diagnosis, triage,
RECIST assessment, patient-management use, or medical-device claims.

## Dataset Contract

Prepared samples are 2D axial slices with image shape `[1, H, W]` for model input. Ground truth uses
half-open `xyxy` pixel boxes on each slice: `0 <= x1 < x2 <= width` and
`0 <= y1 < y2 <= height`. The `source_z` field links every sample back to the canonical repository
volume convention of `z,y,x`.

Positive slices receive boxes only when lesion-mask pixels are present. Negative slices use empty
`boxes` with shape `(0, 4)` and empty labels. Source case ID, synthetic subject ID, timepoint,
scenario, split, spacing, source volume checksum, and source lesion-mask checksum are preserved in
`detection_dataset_manifest.json`.

## Models

The benchmark supports exactly two detector families:

- `faster_rcnn`: a CPU-bounded Faster R-CNN baseline using a small 1-channel CNN backbone and
  torchvision FasterRCNN when torchvision is available. It is configured with `weights=None` and
  `weights_backbone=None`; no pretrained weights are downloaded.
- `tiny_detr`: a native PyTorch DETR-style detector with a small CNN feature extractor, learned
  object queries, transformer decoder, no-object class, and normalized box regression.

Both model families evaluate through the same prediction contract: `boxes`, `labels`, and `scores`.

## Metrics

Detection metrics are box metrics, not segmentation-mask metrics. The evaluator computes 2D box IoU,
score-ordered matching, precision, recall, AP@0.50, AP@0.75, mAP over configured thresholds, false
positives per image, image counts, box counts, and latency summaries.

## Governance Limits

Synthetic 2D boxes do not prove clinical lesion detection. Confidence scores are engineering model
outputs, not clinical probabilities. All outputs require human engineering review and must not be
used for diagnosis, triage, RECIST assessment, patient management, or medical-device release.
