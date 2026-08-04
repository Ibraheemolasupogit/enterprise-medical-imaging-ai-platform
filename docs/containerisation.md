# Containerisation

This platform is a research and engineering demonstrator. Outputs are intended for technical evaluation and human review only and must not be used for clinical diagnosis or patient-management decisions.

Milestone 13 adds local-only containerisation for the governed FastAPI service and the Streamlit reviewer UI. It does not add Kubernetes, cloud deployment, production authentication, registry publication, or clinical operational use.

## Images

Two images are defined:

- `medical-imaging-api` runs the governed FastAPI application.
- `medical-imaging-reviewer-ui` runs the Streamlit reviewer UI and calls the API over the Compose service name `api`.

Both Dockerfiles are multi-stage Python 3.12 builds. Dependencies are resolved into wheels in the builder stage and installed into a slim runtime stage. Runtime stages use non-root UID/GID `10001:10001`, fixed `/app` working directories, exec-form commands, OCI labels, deterministic Python environment variables, and no copied generated model checkpoints.

Container builds intentionally do not install from the developer `pyproject.toml` dependency resolver directly. They first build a runtime wheelhouse from `requirements/container-runtime.txt`, which pins `torch==2.13.0+cpu` for Linux and uses the official PyTorch CPU wheel index at `https://download.pytorch.org/whl/cpu`. The project wheel is then installed with `--no-deps` so Linux builds do not resolve CUDA, NVIDIA, cuDNN, NCCL, Triton, or related GPU packages. The macOS/local developer workflow remains unchanged.

The Dockerfiles copy the container requirements file before source code so the heavy dependency layer can be reused when application source changes. Development and test extras are not installed in runtime images.

## Runtime Filesystem

The Compose configuration uses read-only root filesystems where practical, `tmpfs` for `/tmp`, dropped Linux capabilities, and `no-new-privileges:true`. Writable paths are limited to explicit output mounts:

- API output mount: `/app/outputs`
- Reviewer export mount: `/app/outputs`
- Temporary path: `/tmp`

Configuration, evidence, and checkpoint mounts are read-only. The reviewer UI does not mount checkpoints.

## Local Network

Compose exposes ports only on `127.0.0.1`:

- API: `127.0.0.1:8000`
- Reviewer UI: `127.0.0.1:8501`

The reviewer UI depends on API health and reaches the API at `http://api:8000` inside the named Compose network. Host networking, privileged mode, and Docker socket mounts are prohibited.

## Health And Readiness

The API container health check calls `/health`. Release smoke verification also checks `/ready`, `/version`, service UID/GID, read-only root filesystem behaviour, writable output mounts, reviewer UI health, and reviewer-to-API reachability. The reviewer UI health check uses Streamlit's local health endpoint. If readiness is degraded because model artefacts are expected to be mounted externally, the smoke evidence records that degraded state explicitly instead of reporting full readiness.
