# Demonstration Script

This platform is a research and engineering demonstrator. Outputs are intended for technical
evaluation and human review only and must not be used for clinical diagnosis or patient-management
decisions.

## Prerequisites

- Python 3.12 with development dependencies installed.
- Optional for runtime demonstrations: Docker, Helm, kubectl, kind and Terraform.
- No AWS credentials are required for checkout-safe evidence.
- Do not run `terraform apply`.

## Five-Minute Version

```bash
make demo-fast
```

Expected result: stage-level Make output ending with portfolio evidence validation `PASS` and
demo-fast evidence `PASS`.

Explain:

- The workflow uses deterministic synthetic evidence.
- Kubernetes and AWS checks are static unless optional local runtimes are available.
- The project is not a diagnostic system or clinical deployment.

Fallback: open `reports/generated/portfolio/portfolio_report.md`.

## Ten-Minute Version

```bash
make validate-config
make verify-synthetic-data
make verify-monitoring
make validate-kubernetes-policy
make validate-aws-policy
make build-operations-evidence
make build-portfolio-evidence
make validate-portfolio-evidence
```

Expected outputs: each target reports `PASS` or completes without error. Discuss the status
semantics: `PASS`, `WARN`, `FAIL`, `INCOMPLETE`, `UNAVAILABLE` and `ERROR`.

## Twenty-Minute Technical Version

```bash
make verify-dicom-quality
make verify-preprocessing
make verify-registration
make verify-localisation
make verify-segmentation
make verify-classification
make verify-longitudinal
make verify-monitoring
make validate-containers
make validate-helm
make validate-kubernetes-policy
make terraform-fmt-check
make validate-aws-policy
make build-operations-evidence
make build-portfolio-evidence
make validate-portfolio-evidence
```

Timing depends on local CPU performance because the model milestones run small CPU-compatible
training flows. Use `make demo-fast` if the interview slot is short.

## Cleanup

```bash
make clean-demo
docker compose down --volumes --remove-orphans
make clean-local-kubernetes
```

Cleanup is local only. It must not delete source files, commit changes, push branches, deploy AWS
resources, or run Terraform apply.

## Prohibited Claims

Do not claim diagnosis, clinical decision support, NHS approval, medical-device approval, live cloud
deployment, clinical validation, automated model promotion, automated retraining or automated
rollback.
