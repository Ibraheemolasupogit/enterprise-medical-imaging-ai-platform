ARG PYTHON_IMAGE=python:3.12.8-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder
WORKDIR /build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
COPY requirements/container-runtime.txt ./requirements/container-runtime.txt
RUN python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /wheels -r requirements/container-runtime.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip wheel --no-deps --wheel-dir /wheels .

FROM ${PYTHON_IMAGE} AS runtime
ARG BUILD_DATE=1970-01-01T00:00:00Z
ARG SOURCE_REVISION=unknown
ARG VERSION=0.1.0
LABEL org.opencontainers.image.title="medical-imaging-api" \
      org.opencontainers.image.description="Research-only governed FastAPI service for synthetic medical-imaging AI evidence." \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="local-clean-room-repository"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    MEDICAL_IMAGING_API_CONFIG=/app/config/api.yaml
WORKDIR /app
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/config /app/evidence /app/checkpoints /app/outputs /tmp/matplotlib \
    && chown -R 10001:10001 /app/outputs /tmp/matplotlib
COPY --from=builder /wheels /wheels
COPY requirements/container-runtime.txt /app/requirements/container-runtime.txt
RUN python -m pip install --no-cache-dir --no-index --find-links /wheels -r /app/requirements/container-runtime.txt \
    && python -m pip install --no-cache-dir --no-index --find-links /wheels --no-deps enterprise-medical-imaging-ai-platform \
    && rm -rf /wheels
COPY --chown=10001:10001 config/container/api.yaml /app/config/api.yaml
COPY --chown=10001:10001 config/longitudinal.yaml /app/config/longitudinal.yaml
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]
CMD ["python", "-m", "uvicorn", "medical_imaging_platform.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
