# Runbooks

Runbooks live under `docs/operations/runbooks/` and cover API readiness, reviewer UI availability,
model load failures, checksum mismatch, latency, error rate, Kubernetes rollout failure, image
rollback, model rollback, evidence corruption, secret/configuration retrieval failure, and
monitoring alert escalation.

Every runbook includes detection, immediate checks, containment, recovery, validation, escalation,
rollback, evidence capture, and prohibited actions. Runbooks support human operators; they do not
automate production rollback or model promotion.
