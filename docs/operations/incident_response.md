# Incident Response

Milestone 17 incident response is deterministic and local. Simulations cover API unavailability,
model checkpoint checksum failure, degraded readiness, latency elevation, inference failures,
reviewer UI dependency failure, Kubernetes restart indicators, monitoring drift escalation,
corrupted evidence, and secret/configuration retrieval failure.

Incident lifecycle states are `detected`, `acknowledged`, `investigating`, `contained`,
`recovering`, `resolved`, `post-incident-review-required`, and `closed`. Transitions require actor
metadata. Automatic closure is prohibited.

Incident records preserve trigger, severity, affected component, timeline, detection evidence,
containment, recovery, verification, owner role, status, and next action. They are operational
simulation evidence only.
