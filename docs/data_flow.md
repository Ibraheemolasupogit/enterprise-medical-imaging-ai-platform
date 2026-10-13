# Current data flow

All demonstrated inputs are synthetic. Generated arrays, DICOM, masks, checkpoints and reports stay
outside versioned source artifacts. M23's common evidence envelopes record source/artifact hashes,
actual run references, model/configuration identity, quality outcomes and data roles.

| Stage | Actual input | Output / role |
|---|---|---|
| DICOM QC/preprocessing | Separate deterministic synthetic DICOM series | Technical QC, `[z,y,x]` volume, spacing, intensity/crop metadata |
| Classical baseline | Validated preprocessed directory | Derived transforms/metrics; canonical preprocessing unchanged |
| Registration/localisation | Explicit preprocessed or registered volumes | Transforms / atlas ROIs and preserved quality findings |
| Segmentation preparation | Synthetic volumes and reference lesion masks | Subject-isolated image/mask samples |
| Classification preparation | Synthetic reference lesion/adrenal masks and volumes | Reference-derived crops/labels; manifest records repaired splits if needed |
| Detection preparation | Synthetic volumes and reference masks | Axial slices / reference xyxy boxes, negative examples |
| Embeddings | Prepared classification samples and trained CNN/ViT checkpoints | Stored float32 normalized matrix / ordered provenance |
| Retrieval | Validated index and recorded or explicit policy | Exact ranked cosine results / evaluation / comparison |
| Longitudinal | Explicit previous/current masks and provenance | Measurements, matching, quality propagation, engineering labels |
| Reviewer | API summaries, bounded previews and evidence | Separate decisions and checksummed review exports |

The canonical demo feeds actual predicted masks to longitudinal analysis. It leaves missing
registration evidence visible rather than fabricating an upstream pass. Older verification commands
use reference masks and remain labelled as such. Training/evaluation ground truth is not a model
prediction. No claim is made that classifier crops or detection labels consume U-Net predictions,
or that localisation feeds every model dataset.

[Architecture](architecture/final_end_to_end_architecture.md), [demo](platform_demo.md),
[evidence contract](evidence_contracts.md).
