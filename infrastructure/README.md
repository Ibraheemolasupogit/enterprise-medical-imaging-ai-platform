# Infrastructure

Implemented container definitions are in [docker](../docker/) and [Compose](../docker-compose.yml).
[Helm](../helm/medical-imaging-platform/) supplies a statically assured Kubernetes design and optional
local deployment workflow. [Terraform](../infra/terraform/README.md) describes AWS target state.

See [deployment documentation](../docs/README.md). Static policy checks are not executed workloads;
no live AWS deployment is claimed. Runtime assurance requires evidence for the current image/config.
