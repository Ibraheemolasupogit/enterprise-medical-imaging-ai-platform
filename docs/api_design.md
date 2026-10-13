# API design

The M1–M23 FastAPI research service is created with `create_app(config_path=None)`. It exposes
segmentation, classification, axial detection, longitudinal analysis, exact embeddings search,
configured index selection and read-only evidence review. [Contracts](api_contracts.md).

Explicit configuration arguments take precedence over `MEDICAL_IMAGING_API_CONFIG`; otherwise the
factory reads that environment variable or `config/api.yaml`. `PLATFORM_ENVIRONMENT` and
`PLATFORM_LOG_LEVEL` override the corresponding YAML values through typed validation.

The execution boundary enforces input/evidence root containment, symlink rejection, bounded array
and body sizes, no NumPy pickle, request IDs, sanitized unexpected errors and security headers.
Readiness checks configured paths/index integrity and reports unavailable capabilities separately
from liveness. Model-file hashes are provenance; no trusted expected-model checksum or training
quality approval is enforced at readiness. Shape/checkpoint compatibility is checked at inference.

The unused API fields `require_model_checksums`, `require_quality_pass`, `request_timeout_seconds`
and `maximum_batch_size` have been removed. Legacy configurations containing them fail explicitly;
there is no server-side inference timeout or batching control implied. Reviewer HTTP timeouts remain
real client controls. Evidence review retains its explicit degraded-review policy.

Streamlit is an API consumer and performs no inference. Authentication, PACS/DICOMweb, databases,
persistent audit storage, clinical approval and internet-facing serving remain outside scope.
[Local startup](local_startup.md), [security](security.md), [container design](containerisation.md).
