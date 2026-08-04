# AWS Terraform Target Architecture

This Terraform tree describes a controlled AWS target architecture for the research and engineering demonstrator. It is safe to validate without AWS credentials when using `terraform init -backend=false`, `terraform validate`, and the repository's custom policy checks.

It does not deploy clinical systems, does not expose inference publicly by default, does not configure SageMaker or GPU node groups, and does not include `terraform apply` in normal workflows.

## Structure

- `modules/networking`: VPC, private workload subnets, optional public ingress subnets, route tables, security groups, and optional VPC flow logs.
- `modules/security`: KMS keys, IAM roles, least-privilege workload policies, and Secrets Manager read boundaries.
- `modules/ecr`: private immutable ECR repositories for API and reviewer UI images.
- `modules/storage`: S3 buckets for checkpoints, synthetic/de-identified artefacts, evidence, and audit/control-plane logs.
- `modules/eks`: EKS control plane, CPU-only managed node group, private endpoint defaults, KMS-backed Kubernetes secret encryption, and OIDC provider support.
- `modules/observability`: CloudWatch log groups, alarms, and CloudTrail.
- `environments/dev`: validation-safe composition with no real backend configured.

## Validation

```bash
make terraform-fmt-check
make terraform-init
make terraform-validate
make validate-aws-policy
make scan-terraform
make build-aws-evidence
make validate-aws-evidence
```

`make aws-plan` is optional and operator-driven. It uses `-backend=false` and does not apply resources. AWS credentials are not required for the static evidence path.
