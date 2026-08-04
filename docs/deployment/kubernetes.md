# Kubernetes Deployment Guide

Milestone 15 provides a secure local Helm chart for the API and reviewer UI. It is a research and
engineering demonstrator deployment path only and is not for diagnosis, clinical decision-making, NHS
approval, medical-device claims, or production cloud deployment.

The chart lives at `helm/medical-imaging-platform/` and renders:

- Deployments and internal ClusterIP Services for the API and reviewer UI.
- ConfigMap values for synthetic/non-clinical runtime configuration.
- Optional Secret references without embedding real secrets.
- ServiceAccount with token automount disabled by default.
- NetworkPolicy, PodDisruptionBudget and HorizontalPodAutoscaler resources.

Use immutable local image tags or digests. The default tag is `0.1.0-m13-local`; do not use
`latest`.

For kind or minikube, load locally built images into the cluster before installing the chart. Example
placeholder commands:

```bash
kind load docker-image medical-imaging-api:0.1.0-m13-local
kind load docker-image medical-imaging-reviewer-ui:0.1.0-m13-local
helm upgrade --install medical-imaging-platform helm/medical-imaging-platform
```

Do not place real secrets in `values.yaml`. Create a Kubernetes Secret outside this repository and
reference its name through `api.secretRefs` or `reviewerUi.secretRefs` when a governed local test
requires sensitive configuration.
