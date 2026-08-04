# Local Release Assurance

This platform is a research and engineering demonstrator. Outputs are intended for technical evaluation and human review only and must not be used for clinical diagnosis or patient-management decisions.

Milestone 13 release assurance is local engineering evidence only. It does not publish images, sign artefacts, certify clinical safety, or establish production deployment readiness.

## Gates

The release workflow validates:

- Container configuration.
- Dockerfile static policy.
- Docker Compose security controls.
- Repository secret scanning with Gitleaks.
- Dependency vulnerability scanning with pip-audit.
- Local image build where Docker is available.
- SBOM generation with Syft for the API and reviewer UI images.
- Image scanning with Trivy for the API and reviewer UI images.
- Compose smoke tests where Docker is available.
- Release evidence checksums.

External scanner absence is reported explicitly as `UNAVAILABLE`; results are not fabricated.
`build-release-evidence` consumes the latest persisted scanner, SBOM, image-scan, and smoke-test
evidence instead of rerunning those commands.

Mandatory release evidence cannot produce an overall `PASS` while image builds, Gitleaks,
pip-audit, Syft SBOM generation, Trivy image scanning, or container smoke evidence are missing,
unavailable, errored, or failed. Hadolint is optional advisory evidence because the internal
Dockerfile validator is the mandatory Dockerfile lint gate. The aggregate release status uses
`PASS`, `FAIL`, `INCOMPLETE`, `UNAVAILABLE`, and `ERROR`; `INCOMPLETE` is the expected status when
required scanner or smoke evidence has not been produced locally.

## Commands

Static checks:

```bash
make validate-containers
make lint-dockerfiles
make scan-secrets
make scan-dependencies
```

Docker-dependent local checks:

```bash
make build-images
make generate-sbom
make scan-images
make container-smoke
make build-release-evidence
make validate-release-evidence
```

`make verify-release` runs the full local release sequence and never pushes images.

## Scanner Bootstrap

Scanner tools are installed on the host only, never inside runtime images. Recommended local bootstrap commands:

```bash
python3 -m pip install "pip-audit>=2.7,<3"
brew install syft
brew install trivy
brew install hadolint
```

Trivy uses its local cache directory when available and the release wrapper sets `--cache-dir .trivy-cache`. If Trivy cannot update or read its vulnerability database, the image scan is recorded as `ERROR` or `UNAVAILABLE` rather than as a security pass.

Hadolint is host-only optional advisory tooling. Its absence is recorded as `UNAVAILABLE` and does
not replace or weaken the mandatory internal Dockerfile checks.

## Limitations

Local release evidence is not a production release, medical-device technical file, clinical safety case, or internet-facing deployment assessment. Generated evidence must remain under ignored repository paths.
