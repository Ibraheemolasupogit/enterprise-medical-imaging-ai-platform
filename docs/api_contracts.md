# API contracts

The local research API returns JSON, request IDs and research-only disclaimers. OpenAPI at `/docs`
is the complete typed request schema when enabled. [Design and configuration](api_design.md).

| Route | Input / output |
|---|---|
| `GET /health` | Process liveness, distinct from inference availability |
| `GET /version` | Package/API versions |
| `GET /ready` | Configured artifact findings, capability availability and degraded status; failures return 503 |
| `POST /v1/segmentation/predict` | Exactly one governed `input_path` or 3D `array`, positive `spacing_mm`, optional policy-controlled `threshold`, `persist_output`, `preview_slice_index`, `source_provenance`; probability/volume summary, warnings, full-mask binding and bounded axial mask preview |
| `POST /v1/classification/predict` | Governed path or array; raw/calibrated probability, threshold, engineering label, abstention and artifact hashes |
| `POST /v1/detection/predict` | Governed path or 2D array; boxes, labels, scores, encoder/checkpoint provenance, warnings and optional persisted output |
| `POST /v1/longitudinal/analyse` | Previous/current mask paths or arrays, spacing, research identifiers, timepoints and upstream statuses; measurements, matching, engineering labels and quality propagation |
| `GET /v1/embeddings/indexes` | Explicit configured indexes, availability, encoder and query sample choices |
| `POST /v1/embeddings/search` | Governed `index_directory`, recorded `sample_id`, integer `top_k` (1–100); ranked cosine scores, query/policy provenance, index/checkpoint hashes and warnings |
| `GET /v1/review/segmentation/{experiment_id}` | Validated bounded segmentation evidence summary |
| `GET /v1/review/classification/{experiment_id}` | Validated bounded classification evidence summary |
| `GET /v1/review/detection/{experiment_id}` | Validated bounded detection evidence summary |
| `GET /v1/review/longitudinal/{analysis_id}` | Validated bounded longitudinal evidence summary |

Prediction paths and index choices are subject to root allowlists. Read-only experiment review omits
raw arrays/model weights. Local embedding provenance includes governed server-side artifact paths;
these responses must not be treated as sanitized public release manifests.

Segmentation bindings hash the full predicted array; the displayed slice is a preview. Outputs are
persisted only when requested. Review decisions/export remain separate from model outputs and are
not clinical acceptance. Embedding-space similarity is not diagnostic similarity.

Errors use bounded API messages: malformed request/validation (400/422), policy containment (403),
missing artifacts (404/503), size limit (413), integrity/compatibility (409), sanitized internal
failure (500). Detection/embeddings retain their task-specific error codes in OpenAPI/runtime.
The `/metrics` endpoint is opt-in and accepts `X-Metrics-Token` when configured.
