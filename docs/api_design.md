# API Design

Milestone 11 implements a local governed FastAPI research interface over existing synthetic
segmentation, classification, longitudinal analysis, and evidence-review capabilities.

The API is intentionally narrow:

- It is created through `medical_imaging_platform.api.app.create_app(config_path=None)`.
- It loads typed settings from `config/api.yaml`.
- It runs locally by default on `127.0.0.1`.
- It accepts `.npy` paths only from configured local roots or compact JSON array payloads.
- It returns summaries, checksums, quality findings, and research disclaimers rather than full image
  volumes or model weights.
- It exposes read-only review endpoints for validated evidence directories.

Implemented controls:

- Request ID propagation through `X-Request-ID`.
- Maximum request-body and array-size limits.
- Configured input and evidence root allowlists.
- Path traversal and symlink blocking.
- NumPy loading with `allow_pickle=False`.
- Deterministic API error codes.
- Sanitized internal error messages.
- `nosniff`, frame-denial, no-store cache, and restrictive content-security headers.
- Readiness checks for configured model and governance artefacts.

Not implemented in Milestone 11:

- Authentication or authorization.
- Internet-facing serving.
- Dashboard or human-review UI.
- DICOMweb, PACS, databases, queues, object storage, or audit-event persistence.
- Docker, Kubernetes, AWS, MLflow, or model registry integration.
- Clinical diagnosis, RECIST, treatment-response assessment, or approved medical-device behavior.

The service is an engineering demonstrator for local research workflows only.
