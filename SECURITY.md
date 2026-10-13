# Security

This M1–M23 platform is a local research and engineering demonstrator. Imaging pipelines, models,
API/reviewer interfaces and evidence controls are implemented. Synthetic smoke execution does not
establish clinical safety or production deployment readiness.

Do not report vulnerabilities publicly when reports could expose sensitive details. Provide a
minimal synthetic reproducer to the repository maintainers through a private reporting channel.

Do not commit patient data, identifying DICOM content, credentials, tokens, restricted assets,
generated images or model checkpoints. Metadata de-identification does not remove burned-in pixels.

Authentication, production authorization, TLS termination and an authenticated persistent audit
store are outside the implemented local service. Kubernetes is statically assured; AWS is target
state. See [security controls](docs/security.md) and [limitations](docs/limitations.md).
