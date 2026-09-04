# Deployment guide

This guide describes the deployment assets that are actually present in this repository. The
primary production path is a source build supervised by systemd. Docker Compose is used for
PostgreSQL and, when enabled, Milvus. SightIndex does not currently include an application
Dockerfile or an all-in-one Compose stack.

## Deployment profile

The recommended baseline is one Linux host with the API and console bound to loopback behind a TLS
reverse proxy. PostgreSQL is the canonical database. Milvus and GPU model services are optional.

SightIndex's API process owns stream-capture threads and the vector-index queue. Run exactly one
Uvicorn worker. Multiple API workers can capture the same stream and process the same queue work.

### Port map

| Component | Container/process port | Recommended host bind | Public exposure |
| --- | ---: | --- | --- |
| SightIndex API + console | `8000` by default | `127.0.0.1:8000` | Through a TLS reverse proxy only |
| Vite development server | `5173` | `127.0.0.1:5173` | Never |
| PostgreSQL | `5432` | `127.0.0.1:5432` | Never |
| Milvus gRPC | `19530` | `127.0.0.1:19530` | Never |
| Milvus health/metrics | `9091` | `127.0.0.1:9091` | Never |
| Visual embedding service | `18021` | `127.0.0.1:18021` | Never |
| Qwen visual reranker | `18022` | `127.0.0.1:18022` | Never |
| SapiensID ReID service | `18031` | `127.0.0.1:18031` | Never |
| Optional external YOLO service | `19121` | `127.0.0.1:19121` | Never |

The Milvus Compose file does not publish its etcd or MinIO ports. Keep that boundary intact.

## Prerequisites

- A recent Linux distribution with systemd.
- Python 3.11 or newer.
- Node.js 22.18 or newer and npm.
- Git, curl, and a C/C++ build toolchain.
- Docker Engine and Docker Compose v2 for PostgreSQL and Milvus.
- Enough storage for the database, source media, generated frames/crops, and backups.
- For NVIDIA inference: a driver/runtime combination supported by the selected model stack.
- For Jetson: a CUDA-enabled PyTorch build compatible with the installed JetPack release.

The commands below use these paths and identities:

```text
Application:  /opt/sightindex
Service user: sightindex
Runtime data: /var/lib/sightindex/data
```

The systemd templates under `deploy/systemd/` use the same values.

## 1. Install the source

Create a dedicated, non-login service account and an empty application directory:

```bash
sudo useradd --system --create-home --home-dir /var/lib/sightindex \
  --shell /usr/sbin/nologin sightindex
sudo install -d -o sightindex -g sightindex /opt/sightindex
sudo install -d -o sightindex -g sightindex /var/lib/sightindex/data
```

Clone a reviewed release commit or tag. Do not deploy by copying a selection of changed files:

```bash
sudo -u sightindex -H git clone https://github.com/otterview-labs/SightIndex.git /opt/sightindex
cd /opt/sightindex
git rev-parse HEAD
```

Record that commit in the release ticket or deployment log.

## 2. Create the Python environment and frontend bundle

```bash
cd /opt/sightindex
sudo -u sightindex -H python3 -m venv .venv
sudo -u sightindex -H .venv/bin/python -m pip install --upgrade pip
sudo -u sightindex -H .venv/bin/python -m pip install -r requirements.txt
sudo -u sightindex -H npm --prefix frontend ci
sudo -u sightindex -H npm --prefix frontend run build
test -f frontend/dist/index.html
```

FastAPI serves `frontend/dist/`; there is no separate frontend service in production.

`requirements.txt` installs the normal API runtime. Optional visual and GPU paths have additional
requirements files. Install only the profile needed by the host:

```bash
# Local visual embedding and model tooling
sudo -u sightindex -H .venv/bin/python -m pip install -r requirements.visual.txt

# Jetson/AGX non-PyTorch dependencies
sudo -u sightindex -H .venv/bin/python -m pip install -r requirements.agx.txt
```

`requirements.agx.txt` intentionally does not install PyTorch or torchvision. On Jetson, provide a
JetPack-compatible PyTorch environment and expose it with `REID_SERVICE_PYTHONPATH` or
`QWEN3_VL_EMBEDDING_PYTHONPATH`. There is no portable command that can select the correct Jetson
wheel without knowing the JetPack release.

## 3. Own the environment file

Create the runtime file from the tracked template:

```bash
cd /opt/sightindex
sudo -u sightindex install -m 600 .env.example .env
sudo -u sightindex sh -c '${EDITOR:-vi} .env'
```

The deployment operator or secret manager owns `.env`; Git does not. At minimum, review:

```dotenv
ENVIRONMENT=production
APP_HOST=127.0.0.1
APP_PORT=8000
PUBLIC_BASE_URL=https://sightindex.example.com
DATA_DIR=/var/lib/sightindex/data

DATABASE_URL=postgresql+psycopg://sightindex:REPLACE_WITH_URL_SAFE_PASSWORD@127.0.0.1:5432/sightindex
POSTGRES_DB=sightindex
POSTGRES_USER=sightindex
POSTGRES_PASSWORD=REPLACE_WITH_A_RANDOM_SECRET

APP_BASIC_AUTH_USERNAME=operator
APP_BASIC_AUTH_PASSWORD=REPLACE_WITH_A_RANDOM_SECRET
```

Use a randomly generated, URL-safe database password or percent-encode it in `DATABASE_URL`. Never
commit the resulting file. If a reverse proxy provides stronger identity-aware access, keep the
application port on loopback and document which layer owns authentication.

The example configuration keeps optional providers disabled. Enable each one only after its
service, model, dimension, and credentials are ready.

## 4. Start PostgreSQL

The root Compose file runs PostgreSQL only. It reads `POSTGRES_DB`, `POSTGRES_USER`, and
`POSTGRES_PASSWORD` from `.env` and binds to loopback by default.

```bash
cd /opt/sightindex
sudo docker compose --env-file .env config --quiet
sudo docker compose --env-file .env up -d postgres
sudo docker compose ps
```

Wait for the health check:

```bash
set -a
. ./.env
set +a
until sudo docker compose exec -T postgres \
  pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do sleep 2; done
```

Do not use the example password in a real deployment.

## 5. Start Milvus when vector search is required

Milvus is optional for the baseline API, but required for the Milvus-backed visual and ReID index
paths. Configure strong MinIO credentials in `.env`, then validate and start the stack:

```dotenv
MINIO_ROOT_USER=sightindex
MINIO_ROOT_PASSWORD=REPLACE_WITH_A_RANDOM_SECRET
MILVUS_ENABLED=true
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_COLLECTION_PREFIX=sightindex
MILVUS_NAMESPACE_ID=production
```

```bash
cd /opt/sightindex
sudo docker compose --env-file .env -f deploy/milvus/docker-compose.yml config --quiet
sudo docker compose --env-file .env -f deploy/milvus/docker-compose.yml up -d
until curl --fail --silent http://127.0.0.1:9091/healthz; do sleep 3; done
sudo -u sightindex -H sh -c 'cd /opt/sightindex && set -a && . ./.env && set +a && .venv/bin/python scripts/check_milvus.py'
```

`scripts/check_milvus.py` writes, searches, validates, and removes a temporary vector. The HTTP
health endpoint alone does not prove that the configured collection path works.

Keep `MILVUS_NAMESPACE_ID` stable when moving the same logical database between hosts. Use a new
visual collection prefix when the model or vector dimension changes.

## 6. Optional model services

Start optional dependencies before the API so readiness can be checked independently.

### SapiensID ReID

The large SapiensID checkpoint is not included in Git. Follow
[`deploy/agx/reid_service/README.md`](../deploy/agx/reid_service/README.md), review its
non-commercial upstream license, and place the required assets before startup.

Relevant settings:

```dotenv
REID_ENABLED=true
REID_SERVICE_URL=http://127.0.0.1:18031
REID_SERVICE_PORT=18031
REID_SERVICE_API_KEY=REPLACE_WITH_A_RANDOM_SECRET
REID_CHECKPOINT_DIR=/var/lib/sightindex/models/sapiensid_wb12m
REID_CHECKPOINT_REVISION=sha256:REPLACE_WITH_THE_READY_RESPONSE_VALUE
```

Install and start the unit only after every required asset exists:

```bash
sudo install -m 644 deploy/systemd/sightindex-reid.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sightindex-reid
curl --fail http://127.0.0.1:18031/health
curl --fail http://127.0.0.1:18031/ready
```

`/health` is liveness. `/ready` returns `503` until the model is loaded and its asset identity is
available. Match the API's expected checkpoint revision to the exact `/ready` response.

### Visual embedding service

The optional systemd unit runs a local embedding worker on `127.0.0.1:18021`. Its provider, model,
dimension, device, and API key come from `.env`:

```bash
sudo install -m 644 deploy/systemd/sightindex-embedding.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sightindex-embedding
curl --fail http://127.0.0.1:18021/health
```

### External YOLO and Qwen reranker helpers

`deploy/agx/start_yolo_service.sh` and `start_qwen3_vl_reranker_gpu.sh` require an
operator-supplied NVIDIA container image, workspace/runtime, and model path. The repository does
not build those images. The scripts fail early when the required values are absent; configure and
test them as separate deployment artifacts before enabling their URLs in the API.

The default Qwen reranker port is `18022`, separate from ReID's `18031`.

## 7. Install and start the API service

Install the unit after the database and any enabled model services are ready:

```bash
cd /opt/sightindex
sudo install -m 644 deploy/systemd/sightindex-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sightindex-api
sudo systemctl status --no-pager sightindex-api
```

The unit reads `APP_HOST` and `APP_PORT` from `.env`, writes logs to the system journal, and runs as
the `sightindex` user.

If background attribute backfill is part of the deployment, install its separate unit only after
the API is healthy:

```bash
sudo install -m 644 deploy/systemd/sightindex-attribute-backfill.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sightindex-attribute-backfill
```

## 8. Verify the deployment

Check the service and its recent logs:

```bash
sudo systemctl is-active sightindex-api
sudo journalctl -u sightindex-api -n 100 --no-pager
curl --fail http://127.0.0.1:8000/health
```

`/health` proves only that the HTTP process is alive. Exercise a database-backed API as the
business smoke test:

```bash
curl --fail --user 'operator:YOUR_PASSWORD' \
  http://127.0.0.1:8000/api/media/counts
```

If application Basic Auth is disabled behind a trusted local proxy, omit `--user`.

For ReID, a `200` from the API status route is not enough; inspect the JSON fields:

```bash
curl --fail --user 'operator:YOUR_PASSWORD' \
  http://127.0.0.1:8000/api/reid/status
```

Confirm that `enabled` and `ready` match the intended deployment and that backlog/index coverage
are plausible. Also call the ReID service's `/ready` endpoint directly from the host.

The final external check should use the TLS URL through the reverse proxy and a non-administrator
client account.

## Data and backup inventory

Back up every state owner together:

| State | Default owner |
| --- | --- |
| Metadata | PostgreSQL volume or local `sightindex.db` for SQLite |
| Media | `DATA_DIR`: uploads, videos, frames, crops, thumbnails, diagnostics |
| Milvus vectors | `milvus-etcd`, `milvus-minio`, and `milvus-data` volumes |
| Model assets | Operator-managed model directories and caches |
| Runtime configuration | Secret-managed `.env` backup, not Git |

The database is the source of truth for metadata; Milvus is a rebuildable index only when all
source records and model identities are retained.

## Upgrade procedure

This project currently uses automatic table creation and additive compatibility migrations rather
than Alembic. A code rollback is not necessarily a schema rollback. Stop writers and take backups
before changing versions.

1. Record the current commit and image/model revisions.
2. Stop the API and optional workers.
3. Back up PostgreSQL (or the SQLite file), `DATA_DIR`, and any non-rebuildable Milvus state.
4. Fetch and check out the reviewed release commit.
5. Reinstall pinned dependencies and rebuild `frontend/dist`.
6. Validate configuration and run tests.
7. Start dependencies, model services, then the API.
8. Run liveness, database, vector, and external smoke checks.

Example database backup:

```bash
cd /opt/sightindex
PREVIOUS_COMMIT=$(git rev-parse HEAD)
BACKUP_DIR=/var/lib/sightindex/backups
DATABASE_BACKUP="$BACKUP_DIR/sightindex-${PREVIOUS_COMMIT}.dump"
DATA_BACKUP="$BACKUP_DIR/data-${PREVIOUS_COMMIT}.tar.gz"
set -a
. ./.env
set +a
sudo systemctl stop sightindex-api sightindex-attribute-backfill 2>/dev/null || true
sudo install -d -m 700 -o root -g root "$BACKUP_DIR"
sudo install -m 600 /dev/null "$DATABASE_BACKUP"
sudo install -m 600 /dev/null "$DATA_BACKUP"
sudo docker compose exec -T postgres \
  pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc \
  | sudo tee "$DATABASE_BACKUP" >/dev/null
sudo tar -C /var/lib/sightindex -czf "$DATA_BACKUP" data
```

Before running those commands, ensure the destination filesystem has enough free space and protect
the backup files as sensitive data.

## Rollback

1. Stop the API and workers.
2. Check out the previously recorded commit.
3. Restore its Python dependencies and rebuild its frontend.
4. Restore the matching database/data backup if the newer release changed persisted state.
5. Restore matching model revisions and vector namespace settings.
6. Start dependencies, optional model services, and the API in that order.
7. Repeat all smoke checks.

Never point old code at a database that has undergone an incompatible forward-only change merely
because the old process starts successfully.

## Security checklist

- Keep the API on loopback unless a firewall and authentication boundary are explicit.
- Put TLS and user authentication in front of any non-local access.
- Use strong, distinct secrets for PostgreSQL, Basic Auth, MinIO, and model-service APIs.
- Do not expose PostgreSQL, Milvus, MinIO, ReID, embedding, reranker, or YOLO ports publicly.
- Treat RTSP URLs, face embeddings, person crops, and model outputs as sensitive data.
- Limit retention with `MEDIA_RETENTION_DAYS` and test cleanup in dry-run mode first.
- Restrict `.env`, database backups, media directories, and model caches to the service account.
- Do not log tokens, full RTSP URLs, or raw biometric payloads.
- Review third-party model licenses before commercial or biometric use.
