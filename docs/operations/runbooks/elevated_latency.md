# Elevated Latency Runbook

## Detection
Latency SLO or CloudWatch-style alarm threshold is breached.
## Immediate Checks
Check request volume, pod restarts, CPU, memory, model load state, and API dependency status.
## Containment
Reduce nonessential requests and pause release activities.
## Recovery
Restore resources or roll back recent configuration with approval.
## Validation
Re-evaluate SLOs and metrics.
## Escalation
Escalate to platform engineering.
## Rollback
Use approved prior image or configuration if a recent change caused latency.
## Evidence Capture
Capture metrics window, request IDs, and operator actions.
## Prohibited Actions
Do not loosen request limits or expose public endpoints.
