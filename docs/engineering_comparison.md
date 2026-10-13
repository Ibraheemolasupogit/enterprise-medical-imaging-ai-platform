# Engineering model comparison

Run `make platform-demo` to generate `reports/generated/platform-demo/engineering_comparison.md`
and its JSON counterpart from actual one-epoch CPU benchmark evidence. No numeric performance claim
is embedded in this document: generated values remain tied to their run, data and machine context.

| Comparison | Measured evidence | Engineering trade-off |
|---|---|---|
| Small 3D CNN / Tiny 3D ViT | Classification metrics, parameter count, checkpoint bytes, training time and latency summary | Local convolutional bias versus patch/token attention; neither is universally superior |
| Faster R-CNN / Tiny DETR | AP50/AP75, configured mAP, precision/recall, false positives per image, parameters, checkpoint bytes and latency | Proposal/RoI processing versus query-based prediction; both use tiny research architectures |
| CNN / ViT embeddings | Recall/precision/mAP at configured K, MRR, consistency, dimensions, matrix bytes and query latency | Representation dimension/storage and exact cosine quality depend on the trained encoder |

The artifact includes cases, prepared sample counts, CPU/thread setting, Python/PyTorch/platform
versions, seed and epoch count. Classification/detection rows include the source manifest/config and
checkpoint checksums. The retrieval benchmark records identical gallery/query membership and actual
K values. Checkpoints retain the classification resource measurements in the same comparison artifact.

Synthetic one-epoch runs test engineering behaviour. They do not justify clinical accuracy claims,
model selection for real studies or production latency estimates. Null metrics remain unavailable;
results are not fabricated. Comparison is under a shared policy within each task family, not a claim
that every family used identical splits or inputs.
