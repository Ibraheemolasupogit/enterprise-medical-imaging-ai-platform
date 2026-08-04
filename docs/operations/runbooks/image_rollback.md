# Image Rollback Runbook

## Detection
Container image release or runtime evidence fails after an image change.
## Immediate Checks
Check immutable tag, SBOM, vulnerability scan, smoke result, and release manifest.
## Containment
Stop image promotion and preserve release evidence.
## Recovery
Select previous immutable image tag with approval.
## Validation
Run container smoke, Kubernetes smoke if available, and evidence validation.
## Escalation
Escalate to platform engineering and security for vulnerabilities.
## Rollback
Reference previous image tag or digest; do not use `latest`.
## Evidence Capture
Capture image metadata, digest, SBOM status, and approval ticket.
## Prohibited Actions
Do not rebuild a different image under the same tag.
