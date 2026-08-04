# Checksum Mismatch Runbook

## Detection
Evidence, model, config, or release checksum validation fails.
## Immediate Checks
Compare expected and observed checksums and identify the source artefact.
## Containment
Stop downstream use of the affected artefact.
## Recovery
Restore from versioned evidence or approved artefact source.
## Validation
Recompute checksums and rebuild evidence.
## Escalation
Escalate to governance and security if tampering is suspected.
## Rollback
Use the last approved checksum manifest.
## Evidence Capture
Capture mismatch details without raw payloads or patient data.
## Prohibited Actions
Do not edit evidence to match corrupted artefacts.
