# API Not Ready Runbook

## Detection
Readiness endpoint returns degraded or not-ready.
## Immediate Checks
Inspect readiness findings, model artefact paths, checksums, and configuration status.
## Containment
Stop promotion or release activity and preserve current evidence.
## Recovery
Restore required model/config artefacts or operate in documented degraded review mode.
## Validation
Run readiness, metrics, and operations evidence validation.
## Escalation
Escalate to platform engineering and model governance if artefacts are missing.
## Rollback
Use approved previous configuration or model-version reference only.
## Evidence Capture
Capture request ID, correlation ID, findings, checksums, and operator actions.
## Prohibited Actions
Do not bypass readiness, fabricate checksums, or use clinical data.
