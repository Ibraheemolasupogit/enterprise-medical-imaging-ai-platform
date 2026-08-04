# Monitoring Alert Escalation Runbook

## Detection
Monitoring, drift, SLO, or operations evidence returns WARN or ALERT.
## Immediate Checks
Review alert source, affected component, model version, and evidence checksums.
## Containment
Pause automated claims and preserve evidence.
## Recovery
Follow component-specific runbook and document next action.
## Validation
Re-run monitoring or operations evidence validation.
## Escalation
Escalate to platform engineering, governance, or security based on source.
## Rollback
Consider rollback only with explicit operator approval.
## Evidence Capture
Capture alert summary, incident ID, and reviewer actions.
## Prohibited Actions
Do not claim clinical performance deterioration from synthetic monitoring.
