# Failed Model Load Runbook

## Detection
Startup, readiness, or inference logs report model-loading failure.
## Immediate Checks
Verify checkpoint path, checksum, model type, framework version, and registry status.
## Containment
Block governed inference and preserve failed-load logs.
## Recovery
Restore approved checkpoint or retire the affected model version.
## Validation
Run readiness and model registry validation.
## Escalation
Escalate to model governance.
## Rollback
Use a previous approved model version with explicit approval metadata.
## Evidence Capture
Capture checkpoint checksum, config checksum, registry entry, and incident record.
## Prohibited Actions
Do not load unapproved weights or alter registry state without approval.
