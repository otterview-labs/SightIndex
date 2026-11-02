FROM node:22-bookworm-slim AS frontend
WORKDIR /build
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
ENV VITE_PUBLIC_DEMO=true
RUN npm run build

FROM python:3.12-slim-bookworm
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="SightIndex sample-only demo" \
      org.opencontainers.image.revision="${SOURCE_REVISION}"
COPY deploy/demo/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && pip check \
    && groupadd --gid 1000 sightindex \
    && useradd --uid 1000 --gid 1000 --create-home sightindex
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /opt/sightindex
COPY app/ app/
COPY main.py ./
COPY samples/nasa-walk.jpg samples/nasa-walk.jpg
COPY samples/demo/ samples/demo/
COPY --from=frontend /build/dist/ frontend/dist/
COPY deploy/demo/start.sh /opt/sightindex/start-demo.sh
RUN python -m pip freeze > /opt/sightindex/image-requirements.txt
USER sightindex
EXPOSE 8000
CMD ["sh", "/opt/sightindex/start-demo.sh"]
