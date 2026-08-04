# Reviewer UI Unavailable Runbook

## Detection
Reviewer UI health or API-call metrics fail.
## Immediate Checks
Check UI process, API base URL, timeout, retry exhaustion, and circuit state.
## Containment
Pause reviewer workflow and preserve session evidence.
## Recovery
Restart the UI process or restore approved configuration.
## Validation
Confirm UI health and API connectivity.
## Escalation
Escalate to platform engineering if API connectivity remains unavailable.
## Rollback
Use previous immutable UI image or reviewed config with approval.
## Evidence Capture
Capture UI request IDs, API errors, and configuration checksum.
## Prohibited Actions
Do not point the UI at an unapproved or public API.
