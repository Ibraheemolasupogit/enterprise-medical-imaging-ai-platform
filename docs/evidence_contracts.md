# Common evidence and review contracts

Existing task-specific validators remain authoritative. M23 adds a small `EvidenceEnvelope` sidecar
rather than replacing manifests across the platform. The canonical demo adopts it across preprocessing,
classical baselines, localisation, segmentation, classification, detection, longitudinal analysis,
embeddings and benchmark artifacts. Native embedding generation and segmentation inference also emit sidecars.
API review uses the matching `ArtifactBinding` identity contract.

| Domain | Existing primary contract | Common-envelope interpretation |
|---|---|---|
| Preprocessing | metadata, volume/report hashes and geometry | Derived volume/metadata, source DICOM reference |
| Classical baseline | summary/metrics and checksum manifest | Derived comparison; canonical source unchanged |
| Localisation | localisation JSON and ROI checksums | Derived atlas output; preserve quality status |
| Segmentation | model manifest, state dictionaries, inference arrays | Model evidence or predicted mask |
| Classification | dataset/model/benchmark manifests | Reference-derived labels and trained model evidence |
| Detection | dataset/model/benchmark manifests | Reference-derived boxes and trained detector evidence |
| Longitudinal | pair manifest and measurements | Explicit predicted/reference mask role |
| Embeddings/retrieval | matrix, ordered manifest, policy and hashes | Derived index and exact-search comparison |

The envelope records artifact type/reference/hash, run ID, source ID/hash, optional model/checkpoint
and configuration hashes, generation time, quality status, warnings, data role and capability status.
`evidence_checksum` hashes canonical JSON excluding only itself. Sidecars are additive and never alter
model arrays or existing manifest schemas. The demo validates the artifact, source and file references.

A reviewer decision may carry an exact artifact binding. Its review ID incorporates that binding;
changing the artifact changes decision identity. Export rejects a response whose binding differs from
the decision. `verify_artifact` compares an explicitly supplied local artifact against its binding.
Legacy unbound evidence remains readable and is identified as unbound in the UI.

Embedding review binds to the full index manifest hash; the separately displayed index checksum
covers the matrix. The review summary export records the query and ranking.

File hashes cover `.npy` container bytes. Array hashes cover canonical shape/dtype/contiguous array
bytes and are labelled `sha256-array`; these are different hash contracts. Segmentation API previews
bind to the **full** uint8 predicted mask, rather than only the displayed axial slice. Source provenance
uses the float32 inference input. The review export excludes preview pixels and preserves decisions
separately from immutable original predictions.

Checksums detect accidental changes relative to evidence, not authenticity against an attacker who
can replace both artifacts and manifests. Signed release provenance and an authenticated audit store
remain outside the local demonstrator's scope.
