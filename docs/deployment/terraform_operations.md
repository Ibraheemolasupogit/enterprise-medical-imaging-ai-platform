# Terraform Operations

Terraform lives under `infra/terraform/` and is designed for static validation without AWS
credentials. The repository provides no normal `terraform apply` target.

Safe local checks:

```bash
make terraform-fmt-check
make terraform-init
make terraform-validate
make validate-aws-policy
make scan-terraform
make build-aws-evidence
make validate-aws-evidence
```

`make terraform-init` uses `terraform init -backend=false`. A remote backend example exists at
`infra/terraform/environments/dev/backend.tf.example`, but it is intentionally commented out.

`make aws-plan` is optional and operator-driven. It does not apply resources and records
`INCOMPLETE` when AWS credentials are absent. Generated plans must not be treated as approval to
deploy paid AWS resources.

Operational safeguards:

- Review cost drivers before enabling NAT gateways, EKS, load balancers, or longer log retention.
- Keep examples free of real account IDs, credentials, secret values, and patient data.
- Use immutable image tags and private ECR repositories.
- Keep public inference exposure disabled unless a later milestone explicitly designs and approves it.
- Keep model promotion, rollback, and retraining human-governed and outside Terraform automation.
