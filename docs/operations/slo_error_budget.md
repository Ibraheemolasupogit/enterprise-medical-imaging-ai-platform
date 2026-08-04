# SLO And Error Budget

Milestone 17 defines demonstrator-only SLOs for API availability, API readiness, request latency,
inference error rate, reviewer UI availability, recovery time objective, and recovery point
objective for evidence and model artefacts.

Thresholds are synthetic engineering thresholds. They do not claim clinical performance, diagnostic
safety, medical-device reliability, or NHS approval.

Error-budget evaluation uses deterministic synthetic windows and returns `PASS`, `WARN`, or `ALERT`
where appropriate. Burn-rate style results trigger investigation and evidence capture only; they do
not trigger automated retraining, deployment, model promotion, or production rollback.
