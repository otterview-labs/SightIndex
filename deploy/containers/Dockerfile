ARG NODE_IMAGE=node:22-bookworm-slim
ARG TORCH_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

FROM ${NODE_IMAGE} AS frontend
ARG NPM_REGISTRY=https://registry.npmjs.org
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --registry="${NPM_REGISTRY}" --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM ${TORCH_IMAGE} AS runtime
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="SightIndex" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/containers/requirements.txt /tmp/sightindex-requirements.txt
RUN python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" \
      -r /tmp/sightindex-requirements.txt \
    && python -m pip check \
    && python -c "import sys, torch; assert sys.version_info >= (3, 11); assert torch.__version__.split('+')[0] == '2.10.0'" \
    && apt-get purge -y --auto-remove build-essential python3-dev \
    && rm -f /tmp/sightindex-requirements.txt
RUN groupadd --gid 1000 sightindex \
    && useradd --uid 1000 --gid 1000 --create-home sightindex \
    && mkdir -p /opt/sightindex/data /home/sightindex/.cache \
    && chown -R sightindex:sightindex /home/sightindex /opt/sightindex/data
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/sightindex \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    HF_HUB_OFFLINE=1
WORKDIR /opt/sightindex
COPY app/ app/
COPY main.py pyproject.toml ./
COPY deploy/containers/healthcheck.py healthcheck.py
COPY deploy/agx/reid_service/ deploy/agx/reid_service/
COPY --from=frontend /build/dist/ frontend/dist/
RUN python -m pip freeze > /opt/sightindex/image-requirements.txt
USER sightindex
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
