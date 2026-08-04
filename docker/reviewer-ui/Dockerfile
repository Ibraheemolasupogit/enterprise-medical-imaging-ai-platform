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
LABEL org.opencontainers.image.title="medical-imaging-reviewer-ui" \
      org.opencontainers.image.description="Research-only Streamlit reviewer UI for governed synthetic medical-imaging AI evidence." \
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
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/config /app/outputs /tmp/matplotlib /tmp/streamlit \
    && chown -R 10001:10001 /app/outputs /tmp/matplotlib /tmp/streamlit
COPY --from=builder /wheels /wheels
COPY requirements/container-runtime.txt /app/requirements/container-runtime.txt
RUN python -m pip install --no-cache-dir --no-index --find-links /wheels -r /app/requirements/container-runtime.txt \
    && python -m pip install --no-cache-dir --no-index --find-links /wheels --no-deps enterprise-medical-imaging-ai-platform \
    && rm -rf /wheels
COPY --chown=10001:10001 config/container/reviewer_ui.yaml /app/config/reviewer_ui.yaml
USER 10001:10001
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"]
CMD ["medical-imaging-platform", "serve-reviewer-ui", "--config", "/app/config/reviewer_ui.yaml", "--host", "0.0.0.0", "--port", "8501"]
