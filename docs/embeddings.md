# M22 Embeddings and Similarity Retrieval

M22 extracts pre-head representations from the existing small_3d_cnn and tiny_3d_vit classifiers for exact cosine retrieval over validated 3D classification samples. The gallery is train plus validation and queries are test, preserving subject-isolated splits.

Matrices are float32 L2-normalized NumPy arrays with ordered provenance manifests and SHA-256 evidence. Similarity is an embedding-space engineering measure, not diagnostic similarity. This is synthetic/de-identified research only and does not support diagnosis, triage, RECIST, or patient management.

`generate-embeddings` accepts the prepared `classification_manifest.json` and the classification configuration used to train the checkpoint. The legacy `classification_dataset_manifest.json` and flat classification configuration remain supported.

`config/embeddings.yaml` defines `top_k` as an ordered list of unique positive integers; the typed policy and evaluation/benchmark backends use a tuple. `evaluate-retrieval` and `run-embedding-benchmark` load this policy by default (`--config` selects another policy), and `--top-k 1 5` overrides the collection. Unsorted, duplicate, empty, non-integer, and non-positive values are rejected. `query-similar-studies --top-k 5` uses one positive integer.

Generated indexes restrict retrieval candidates to their gallery splits. Evaluation and benchmarking exclude query samples and use train/validation gallery relevance. Benchmark output includes the actual K values, gallery/query membership, source-index manifest hashes, and `evidence/checksum_manifest.json` for the comparison artifacts.

## Governance limitations

Embedding-space similarity is not diagnostic similarity. High similarity does not imply the same disease, and low similarity does not exclude clinically related findings. Synthetic/de-identified data do not establish clinical validity. Embeddings may encode acquisition and preprocessing artifacts and representation bias. False or missed similarity, embedding drift, stale indexes, checkpoint/index incompatibility, subject or duplicate leakage, and identifier leakage are explicit risks. Results require human interpretation and are not for diagnosis, triage, RECIST, or patient-management use.
