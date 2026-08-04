# Kubernetes Rollout Failure Runbook

## Detection
Deployment, pod readiness, or rollout status fails.
## Immediate Checks
Inspect events, image tag, probes, resources, security context, and NetworkPolicy.
## Containment
Stop additional rollout attempts and keep services internal.
## Recovery
Prepare Helm rollback plan with operator approval.
## Validation
Run Kubernetes policy and smoke evidence when local runtime is available.
## Escalation
Escalate to platform engineering.
## Rollback
Use `helm rollback` plan only after approval.
## Evidence Capture
Capture rendered manifests, events, and rollout diagnostics.
## Prohibited Actions
Do not disable security contexts or expose public ingress as a workaround.
