# SightIndex

SightIndex is a self-hosted visual indexing and retrieval service. It accepts images, videos, and
RTSP/HTTP streams; extracts searchable media and person crops; and exposes the results through a
FastAPI API and a Vue console.

The repository is an experimental reference implementation. Face recognition and person
re-identification process biometric data. Deploy them only with a lawful basis, appropriate
consent, access controls, retention limits, and human review.

## What is included

- Image, video, and live-stream ingestion.
- Person detection, crop generation, thumbnails, and counting lines.
- Person, face, visual, and structured-attribute search APIs.
- Optional InsightFace face embeddings.
- Optional Milvus indexes for visual and person ReID vectors.
- Optional SapiensID-based cross-camera candidate retrieval.
- A Vue 3 operator console served by the FastAPI process.
- PostgreSQL for normal deployments and SQLite for local evaluation.

SightIndex returns ranked evidence and confidence metadata. ReID results are candidates, not proof
of identity or a continuous path between cameras.

## Architecture

| Component | Role | Required |
| --- | --- | --- |
| FastAPI + Vue console | API, media ingestion, search, and operator UI | Yes |
| PostgreSQL or SQLite | Metadata and canonical records | Yes |
| Local filesystem | Uploaded media, frames, crops, thumbnails, and models | Yes |
| Milvus | Vector indexes for visual and ReID search | Optional |
| ReID service | SapiensID inference over person crops | Optional |
| Embedding/VLM services | Visual embeddings, reranking, and structured attributes | Optional |

The API process also owns stream-capture threads and the vector indexing queue. Run one API worker
per deployment unless those responsibilities are moved to separate workers.

## Quick start

Prerequisites:

- Python 3.11 or newer.
- Node.js 22.18 or newer for the locked frontend toolchain.
- A C/C++ build toolchain for native Python dependencies.

Clone the public repository and create a local environment:

```bash
git clone https://github.com/otterview-labs/SightIndex.git
cd SightIndex
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.dev.txt
cp .env.example .env
```

The example configuration uses SQLite and disables external model services. Build the console and
start the API:

```bash
npm --prefix frontend ci
npm --prefix frontend run build
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Verify the process from another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/api/media/counts
```

Then open:

- Console: `http://127.0.0.1:8000/`
- OpenAPI: `http://127.0.0.1:8000/docs`

For frontend development, run `npm --prefix frontend run dev`; Vite listens on port `5173` and
proxies API calls to `http://127.0.0.1:8000` by default.

## Deployment

The supported repository assets use a source build managed by systemd, with PostgreSQL and
optionally Milvus managed by Docker Compose. The application itself does not currently ship a
Dockerfile.

See [docs/deployment.md](docs/deployment.md) for:

- host preparation and dependency order;
- environment ownership and secrets;
- PostgreSQL, Milvus, API, and optional GPU services;
- port and network exposure rules;
- health and business-level smoke checks;
- upgrades, backups, and rollback.

Jetson/AGX helper scripts live under `deploy/agx/`. Some optional YOLO and reranker helpers require
an operator-supplied container image and model path; they are integrations, not bundled images.

## Configuration

`.env.example` is the configuration template of record. Its default profile is intentionally small:

- SQLite database in the repository working directory;
- local `data/` storage;
- no Milvus, VLM, visual embedding, or ReID service;
- API bound by the command that starts Uvicorn.

For PostgreSQL development, set a strong `POSTGRES_PASSWORD`, update `DATABASE_URL`, and run:

```bash
docker compose config --quiet
docker compose up -d postgres
```

For Milvus:

```bash
docker compose -f deploy/milvus/docker-compose.yml config --quiet
docker compose -f deploy/milvus/docker-compose.yml up -d
curl --fail http://127.0.0.1:9091/healthz
MILVUS_ENABLED=true .venv/bin/python scripts/check_milvus.py
```

Milvus binds to loopback by default. Do not expose PostgreSQL, Milvus, model services, or raw media
storage directly to an untrusted network.

## Model services

All heavyweight model files are runtime assets and should be downloaded deliberately before
service startup. Do not download models in request handlers.

- InsightFace model packs belong under a configured `FACE_INSIGHTFACE_ROOT`.
- Visual embedding and reranking providers are selected with the corresponding `.env` settings.
- The SapiensID checkpoint is not stored in this repository; see
  [deploy/agx/reid_service/README.md](deploy/agx/reid_service/README.md).
- The lightweight DFA aligner asset inside the vendored SapiensID subset is tracked under its
  upstream non-commercial license.

Keep vector collections separate when models, dimensions, preprocessing versions, or logical
namespaces change.

## API examples

Upload an image:

```bash
curl --fail -X POST http://127.0.0.1:8000/api/images/upload \
  -F 'file=@/path/to/image.jpg'
```

Register a stream using a placeholder URL:

```bash
curl --fail -X POST http://127.0.0.1:8000/api/streams \
  -H 'content-type: application/json' \
  -d '{
    "name": "entrance-camera",
    "stream_url": "rtsp://camera.example.test/stream1",
    "protocol": "rtsp",
    "frame_interval_seconds": 2
  }'
```

Start the registered stream:

```bash
curl --fail -X POST http://127.0.0.1:8000/api/streams/{stream_id}/start
```

Search person crops:

```bash
curl --fail -X POST http://127.0.0.1:8000/api/search/person-crops \
  -H 'content-type: application/json' \
  -d '{"query":"red jacket and backpack","top_k":20,"filters":{}}'
```

Credentials embedded in stream URLs are stored with the stream record. Use a dedicated camera
account, restrict database access, and avoid placing real credentials in shell history or logs.

## Development

Backend checks:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

Frontend checks:

```bash
npm --prefix frontend run typecheck
npm --prefix frontend run build
```

When the API contract changes, start the backend and regenerate/check the frontend schema:

```bash
npm --prefix frontend run gen:api
npm --prefix frontend run check:api
```

## Repository layout

```text
app/                   FastAPI routes, services, models, schemas, and settings
frontend/              Vue 3 console
tests/                 Backend tests
deploy/milvus/         Local Milvus Compose stack
deploy/systemd/        Linux service templates
deploy/agx/            Optional Jetson/AGX and model-service helpers
docs/                  Architecture, API, calibration, and deployment notes
scripts/               Maintenance, import, model, and diagnostic tools
```

Further references:

- [Visual model APIs](docs/visual-ai-apis.md)
- [Cross-camera ReID design](docs/cross-camera-reid.md)
- [ReID walkthrough and calibration](docs/reid-walkthrough-calibration.md)
- [Multimodal retrieval notes](docs/multimodal-retrieval/README.md)
- [Implementation roadmap](docs/implementation-roadmap.md)

## Licensing and third-party code

This repository does not currently grant a project-wide open-source license. Public visibility
allows inspection but does not by itself grant reuse rights.

The vendored SapiensID inference subset is separately licensed under CC BY-NC 4.0. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the license retained beside that source before
using or redistributing it. Model weights may have additional terms.
