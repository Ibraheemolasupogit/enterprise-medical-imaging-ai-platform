# Secret Configuration Retrieval Failure Runbook

## Detection
Runtime secret or configuration reference cannot be retrieved.
## Immediate Checks
Check secret reference name, IAM boundary, KMS permission, and config checksum.
## Containment
Block affected workflow and avoid logging secret values.
## Recovery
Restore approved secret reference or configuration.
## Validation
Run readiness and operations evidence validation.
## Escalation
Escalate to security and platform engineering.
## Rollback
Use previous approved config reference with operator approval.
## Evidence Capture
Capture reference names, not secret values.
## Prohibited Actions
Do not commit secrets or print secret payloads.
