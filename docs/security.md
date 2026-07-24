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
