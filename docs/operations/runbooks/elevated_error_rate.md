# Elevated Error Rate Runbook

## Detection
API 5xx, validation failure, or inference failure rate exceeds threshold.
## Immediate Checks
Inspect structured errors, readiness, model checksums, and recent changes.
## Containment
Pause governed inference for affected route or model version.
## Recovery
Restore dependency, rollback approved config, or retire affected model version.
## Validation
Run normal operations simulation and evidence validation.
## Escalation
Escalate to platform engineering and governance.
## Rollback
Use human-approved image, config, or model rollback path.
## Evidence Capture
Capture status classes, request IDs, correlation IDs, and findings.
## Prohibited Actions
Do not suppress errors or remove abstention/degraded signals.
