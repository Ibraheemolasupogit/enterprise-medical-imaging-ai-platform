# Security

Milestone 1 security controls focus on repository hygiene:

- No real patient data.
- No credentials.
- No cloud account details.
- No generated DICOM or NIfTI files.
- No model artefacts.
- Explicit research-only use boundaries.
- Basic static security scanning with Bandit.
- DICOM working data ignored under `data/dicom/`.
- Private DICOM tags removed by default during de-identification.
- Direct identifier values excluded from normal metadata output and audits.
- In-place source overwrite refused by default.

Future milestones will add controls for de-identification audit, object storage, access control, container scanning, infrastructure validation, and operational monitoring.

Metadata de-identification cannot guarantee removal of burned-in pixel identifiers. Pixel review, OCR, and pixel redaction are not implemented.

Milestone 4 escalates `BurnedInAnnotation=YES`, but absence of that metadata does not guarantee absence of pixel PHI.

Milestone 11 API security controls:

- Localhost binding by default, with `0.0.0.0` blocked unless explicitly allowed in config.
- Configured local input and evidence root allowlists.
- `.npy` input restriction with NumPy `allow_pickle=False`.
- Path traversal, remote URL, and symlink input rejection.
- Request-body and array-size limits.
- Request IDs and sanitized deterministic error payloads.
- Security headers for no-store cache behavior, frame denial, content-type sniffing prevention, and
  restrictive content security policy.
- Read-only evidence review that returns bounded public summaries instead of raw local manifests.

Authentication, authorization, TLS termination, persistent audit logging, rate limiting, network
deployment, and secrets management remain future work.

Milestone 12 reviewer UI security controls:

- Loopback API URL and Streamlit binding by default.
- Remote API endpoints and remote binding rejected unless explicitly configured.
- Bounded `.npy` and JSON upload extension checks.
- Upload-size validation before NumPy processing.
- NumPy loading with `allow_pickle=False`.
- Finite-value and 3D-shape checks for uploaded arrays.
- Safe evidence identifiers and export path containment.
- No arbitrary HTML rendering, shell execution, direct URL fetching, or model-weight display.
- No logging of raw arrays, sensitive payloads, or reviewer notes.
## Container Controls

Milestone 13 adds local container controls: non-root UID/GID `10001:10001`, read-only root
filesystems, dropped Linux capabilities, `no-new-privileges`, local-only port bindings, no Docker
socket mounts, no privileged mode, bounded health checks, `.dockerignore` exclusions for generated
data and secrets, scanner hooks, and ignored release evidence.

## Kubernetes Controls

Milestone 15 adds Helm and Kubernetes controls for the API and reviewer UI:

- Pod security contexts enforce non-root UID/GID `10001`, `runAsNonRoot`, and RuntimeDefault seccomp.
- Container security contexts enforce read-only root filesystems, `allowPrivilegeEscalation=false`,
  `privileged=false`, and `capabilities.drop=["ALL"]`.
- Service account token automount is disabled by default.
- Writable paths use bounded `emptyDir` volumes only.
- Host networking, host PID, host IPC, and hostPath volumes are rejected by static policy checks.
- CPU and memory requests and limits are mandatory.
- API and reviewer UI services default to internal `ClusterIP`; Ingress is disabled by default.
- NetworkPolicy renders default-deny behavior, reviewer-UI-to-API ingress, and DNS egress.
- Real secrets are not included in chart defaults; sensitive settings must use external Secret
  references.

## AWS Infrastructure Controls

Milestone 16 adds AWS target-state controls in Terraform without deploying resources:

- Private ECR repositories use immutable tags, scan-on-push, lifecycle policies, and KMS encryption.
- EKS is private-endpoint by default, uses CPU-only managed nodes, encrypts Kubernetes secrets with
  KMS, enables control-plane logging, and supports OIDC workload identity.
- S3 buckets block public access, require TLS, use KMS encryption, enable versioning, and separate
  checkpoints, synthetic/de-identified artefacts, governed evidence, and CloudTrail logs.
- IAM policies avoid wildcard administrative actions and separate API checkpoint reads from
  monitoring evidence writes.
- Secrets Manager uses explicitly named secret references only; no secret values are committed.
- CloudWatch log retention and alarms are defined, and CloudTrail remains distinct from application
  audit evidence.

## Operations Controls

Milestone 17 adds production-style observability and resilience controls for local demonstration:

- The API `/metrics` endpoint is disabled by default and must be explicitly enabled in
  configuration. If a metrics token is configured, requests without the matching
  `X-Metrics-Token` header are rejected.
- Metrics use bounded engineering labels such as method, route, status and outcome. Patient
  identifiers, free-text payloads, array values, local secret paths and model contents are not valid
  metric labels.
- Structured log events include request and correlation identifiers, route, status, latency, actor
  type and event name while redacting sensitive keys and refusing raw array or payload logging.
- Reviewer UI API calls use bounded retries, short backoff, timeouts and a simple circuit breaker so
  dependency failures degrade safely rather than looping indefinitely.
- Incident, rollback and recovery evidence is synthetic and manual. No automated model promotion,
  retraining, rollback, production paging, public exposure or AWS deployment is implemented.
