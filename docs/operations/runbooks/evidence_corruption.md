# Evidence Corruption Runbook

## Detection
Checksum validation or evidence parser reports corruption.
## Immediate Checks
Identify affected evidence domain and compare checksum manifests.
## Containment
Freeze related release, model, monitoring, or incident decisions.
## Recovery
Restore from versioned local/S3 evidence where available.
## Validation
Rebuild and validate evidence checksums.
## Escalation
Escalate to governance and security if tampering is suspected.
## Rollback
Use previous validated evidence manifest only.
## Evidence Capture
Capture mismatch details and restoration source.
## Prohibited Actions
Do not hand-edit evidence to force a pass.
