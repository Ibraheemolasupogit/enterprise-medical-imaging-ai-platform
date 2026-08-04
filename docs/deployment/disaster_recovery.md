# Disaster Recovery And Rollback

Milestone 16 defines target-state AWS recovery controls but does not deploy live AWS resources.

Planned recovery controls:

- S3 versioning and lifecycle controls for model checkpoints, synthetic/de-identified artefacts,
  governed evidence, and control-plane audit logs.
- KMS customer-managed keys with rotation for EKS secrets, S3 storage, ECR and logs.
- Immutable ECR tags so a previously approved container digest can be selected deliberately.
- CloudTrail log-file validation for AWS control-plane audit trails.
- CloudWatch alarms for readiness failures, API error rate, restart rate, latency and node pressure.
- Terraform evidence and checksum manifests for change review.

Rollback remains a human change-control activity. Terraform does not automatically roll back
infrastructure, Helm does not automatically promote model versions, and monitoring drift evidence does
not trigger automated retraining or deployment.

Application audit evidence and CloudTrail evidence are separate. CloudTrail records AWS control-plane
activity; it does not replace application-level reviewer actions, model registry history, monitoring
evidence, or export/override events.

Milestone 17 adds deterministic rollback and recovery simulation evidence. The simulation records a
human approval reference, rollback target, registry state, checksum verification, recovery checks and
post-incident review requirement. It does not uninstall live workloads, promote model versions,
modify Terraform state, restore patient data, trigger automated rollback, or prove disaster-recovery
readiness for a production clinical service.
