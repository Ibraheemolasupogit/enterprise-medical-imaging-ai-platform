# Kubernetes Operations Guide

Milestone 15 operations are deliberately local and bounded. The normal quality path performs static
Helm and manifest validation without requiring a live cluster.

Run checkout-safe validation:

```bash
make validate-helm
make render-kubernetes
make validate-kubernetes-policy
make build-kubernetes-evidence
make validate-kubernetes-evidence
```

When kind or minikube plus kubectl are available, runtime checks may be recorded separately:

```bash
make deploy-local-kubernetes
make kubernetes-smoke
make clean-local-kubernetes
```

If runtime tooling is unavailable, evidence records `UNAVAILABLE` and the overall deployment evidence
remains `INCOMPLETE`. This is an honest status, not a security pass.

Investigation and rollback are manual. A failed readiness check, policy violation, or smoke failure
must open human review of the Helm values, rendered manifests, image versions, model registry state,
and monitoring evidence. Automated retraining, automatic model promotion, automated rollback, AWS
deployment and public inference exposure are out of scope.

Milestone 17 extends the chart and validation checks with production-style local operations controls:
structured-logging annotations, optional Prometheus scrape annotations, API startup probes, revision
history, topology spread, preferred anti-affinity, resource checks, and degraded-readiness evidence.
These controls support local engineering review only. ServiceMonitor/PodMonitor integration, public
Ingress, service mesh, external alert routing and automated rollback remain disabled or out of
scope unless a future milestone explicitly implements them.
