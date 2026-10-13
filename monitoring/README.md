# Monitoring and operations

The implemented [monitoring module](../src/medical_imaging_platform/governance/monitoring.py) generates local
synthetic registry/drift/audit evidence. [Operations](../docs/operations/observability.md) documents
opt-in API metrics, deterministic incident/recovery simulations and manual response boundaries.

Production monitoring, alert routing and automatic rollback are not implemented. See the
[documentation index](../docs/README.md) for runbooks and deployment design.
