# Observability

Milestone 17 defines local and target-state observability for the research engineering
demonstrator. Observability covers API and reviewer UI request counts, latency, status classes,
inference outcomes, preprocessing/model-loading failures, readiness degradation, UI-to-API failures,
restart indicators, active model version, and request/correlation IDs.

Prometheus-compatible API metrics are exposed through `/metrics` only when enabled by configuration.
The endpoint is disabled by default and supports a static token for protected local/internal use.
Metric labels are bounded to service-defined labels and must not include patient identifiers, raw
DICOM content, image arrays, filenames, paths, request bodies, tokens, secrets, or credentials.

Structured JSON logs include timestamp, severity, service, event type, request ID, correlation ID,
model version where relevant, synthetic/de-identified provenance reference where relevant, and
redacted details. Logs are operational evidence only, not clinical audit approval.
