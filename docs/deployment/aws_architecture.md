# AWS Architecture Guide

Milestone 16 defines a controlled AWS target-state architecture for the existing containerised
Kubernetes platform. It is infrastructure-as-code design and static evidence only; it is not a live
clinical service and does not prove operational or clinical readiness.

The architecture uses private ECR repositories for the API and reviewer UI images, EKS for CPU-only
Kubernetes workloads, S3 for approved model checkpoints, synthetic/de-identified artefacts,
monitoring evidence, application audit evidence, and CloudTrail logs, KMS for encryption, Secrets
Manager references for runtime secrets, CloudWatch for logs, metrics and alarms, CloudTrail for AWS
control-plane auditability, and a VPC with private workload subnets.

Default boundaries:

- Public inference exposure is disabled.
- EKS public endpoint access is disabled.
- NAT gateways are disabled in validation defaults.
- Public ingress subnets are disabled in validation defaults.
- Managed node groups are CPU-only and conservative.
- The reviewer UI is not granted broad direct S3 permissions.
- SageMaker, GPU node groups, automated retraining, automated rollback, automated model promotion,
  production clinical integrations, production DNS, and real TLS certificates are out of scope.

Controlled-access alternatives may include VPN, private bastion patterns, AWS Systems Manager Session
Manager, or private connectivity to the EKS endpoint. Those choices require explicit operator review
and are not enabled by default.

Milestone 14 monitoring evidence maps to CloudWatch application metrics and alarms for engineering
operations only. It must not be interpreted as clinical-performance monitoring or diagnostic safety
surveillance.
