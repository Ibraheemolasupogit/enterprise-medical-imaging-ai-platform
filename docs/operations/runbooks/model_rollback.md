# Model Rollback Runbook

## Detection
Model quality, readiness, monitoring, or incident evidence requires rollback consideration.
## Immediate Checks
Check registry lifecycle, checkpoint checksum, config checksum, and approval metadata.
## Containment
Block affected model version from governed inference.
## Recovery
Approve previous model version through human model change control.
## Validation
Run readiness, registry validation, and monitoring evidence.
## Escalation
Escalate to model governance.
## Rollback
Retire or reject affected version and approve previous version explicitly.
## Evidence Capture
Capture registry manifest, approval ticket, and incident record.
## Prohibited Actions
Do not auto-promote or auto-retire models.
