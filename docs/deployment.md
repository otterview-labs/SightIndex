# Deployment guide

[English](deployment.md) | [简体中文](deployment.zh-CN.md)

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
Application:        /opt/sightindex
Deployment account: sightindex-deploy
Service account:    sightindex
Runtime data:       /var/lib/sightindex/data
```

The deployment account owns the reviewed source, virtual environment, and frontend bundle. The
service account can read those files but can write only runtime paths under `/var/lib/sightindex`.
This prevents a compromised application process from replacing a script that an administrator
will later run with `sudo`. The systemd templates under `deploy/systemd/` use the same paths.

## 1. Install the source

Create separate non-login service and deployment accounts. Both use the `sightindex` group, but
only the deployment account owns the application checkout:

```bash
sudo groupadd --system sightindex
sudo useradd --system --gid sightindex --create-home --home-dir /var/lib/sightindex \
  --shell /usr/sbin/nologin sightindex
sudo useradd --system --gid sightindex --create-home --home-dir /var/lib/sightindex-deploy \
  --shell /usr/sbin/nologin sightindex-deploy
sudo install -d -m 0750 -o sightindex-deploy -g sightindex /opt/sightindex
sudo install -d -m 0750 -o root -g sightindex /var/lib/sightindex
sudo install -d -m 0700 -o sightindex -g sightindex /var/lib/sightindex/data
sudo install -d -m 0700 -o sightindex -g sightindex /var/lib/sightindex/.cache
```

If these identities already exist, verify their home, primary group, and directory ownership
instead of recreating them.

Clone a reviewed release commit or tag. Do not deploy by copying a selection of changed files:

```bash
sudo -u sightindex-deploy -H sh -c 'umask 027; git clone https://github.com/otterview-labs/SightIndex.git /opt/sightindex'
sudo -u sightindex-deploy -H git -C /opt/sightindex rev-parse HEAD
```

Record that commit in the release ticket or deployment log. Confirm that `sightindex` can read the
checkout but cannot modify it:

```bash
sudo -u sightindex test -r /opt/sightindex/main.py
if sudo -u sightindex test -w /opt/sightindex; then
  echo 'unsafe: service account can modify the deployment checkout' >&2
  exit 1
fi
```

## 2. Create the Python environment and frontend bundle

```bash
cd /opt/sightindex
sudo -u sightindex-deploy -H python3 -m venv .venv
sudo -u sightindex-deploy -H .venv/bin/python -m pip install --upgrade pip
sudo -u sightindex-deploy -H .venv/bin/python -m pip install -r requirements.txt
sudo -u sightindex-deploy -H npm --prefix frontend ci
sudo -u sightindex-deploy -H npm --prefix frontend run build
test -f frontend/dist/index.html
```

FastAPI serves `frontend/dist/`; there is no separate frontend service in production.

`requirements.txt` installs the normal API runtime. Optional visual and GPU paths have additional
requirements files. Install only the profile needed by the host:

```bash
# Local visual embedding and model tooling
sudo -u sightindex-deploy -H .venv/bin/python -m pip install -r requirements.visual.txt

# Jetson/AGX non-PyTorch dependencies
sudo -u sightindex-deploy -H .venv/bin/python -m pip install -r requirements.agx.txt
```

`requirements.agx.txt` intentionally does not install PyTorch or torchvision. On Jetson, provide a
JetPack-compatible PyTorch environment and expose it with `REID_SERVICE_PYTHONPATH` or
`QWEN3_VL_EMBEDDING_PYTHONPATH`. There is no portable command that can select the correct Jetson
wheel without knowing the JetPack release.

## 3. Own the environment file

Create the runtime file from the tracked template. It is owned by root and read-only to the service
group; the application process must not be able to change deployment secrets or launcher options:

```bash
cd /opt/sightindex
sudo install -o root -g sightindex -m 0640 .env.example .env
sudoedit .env
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

The environment files used by systemd, Compose, and the deployment helpers must contain simple
`KEY=value` assignments. Quote values containing spaces. Do not place shell commands or variable
substitutions in them.
Use hexadecimal or base64url secrets; the RTX helpers reject shell expansion and command-control
characters instead of trying to reinterpret them.

## Automated RTX 5090 profile

`deploy/rtx5090/` automates the same source-build, systemd, and Milvus path for a single x86_64
host with an NVIDIA RTX 5090. It is a hardware profile, not a second application architecture.
The tracked template defaults to SQLite for a compact single-host deployment. Prefer PostgreSQL
for sustained concurrent ingest, multiple operators, or when PostgreSQL backup and monitoring are
already part of the site platform.

The wrapper deliberately does not run `git pull`. For an update, enter the maintenance window and
stop every process that can lazily import or execute files from the checkout **before** changing
the revision. Record whether the backfill worker was active so it can be resumed explicitly. Once
all writers are confirmed stopped, take the coordinated `DATA_DIR` snapshot and secret-managed
configuration backup, then keep the services stopped through checkout and the wrapper's database
backup:

```bash
sudo -u sightindex-deploy -H git -C /opt/sightindex rev-parse HEAD  # record rollback revision
sudo systemctl is-active sightindex-attribute-backfill.service \
  && BACKFILL_OPTION=--start-backfill || BACKFILL_OPTION=
sudo systemctl disable --now sightindex-embedding.service 2>/dev/null || true
sudo systemctl stop sightindex-attribute-backfill.service \
  sightindex-api.service sightindex-reid.service 2>/dev/null || true
for unit in sightindex-attribute-backfill sightindex-embedding sightindex-api sightindex-reid; do
  if sudo systemctl is-active --quiet "$unit"; then
    echo "failed to stop $unit" >&2
    exit 1
  fi
done
if sudo systemctl is-enabled --quiet sightindex-embedding.service; then
  echo 'failed to disable sightindex-embedding' >&2
  exit 1
fi
# Take the DATA_DIR snapshot here; do not restart any writer afterward.
sudo -u sightindex-deploy -H git -C /opt/sightindex fetch --tags origin
sudo -u sightindex-deploy -H git -C /opt/sightindex checkout --detach <reviewed-commit-or-tag>
sudo -u sightindex-deploy -H git -C /opt/sightindex status --short
sudo -u sightindex-deploy -H git -C /opt/sightindex rev-parse HEAD
```

On the first installation only, create and edit the RTX-specific environment file:

```bash
cd /opt/sightindex
if ! sudo test -e .env; then
  sudo install -o root -g sightindex -m 0640 \
    deploy/rtx5090/sightindex.env.example .env
  sudoedit .env
fi
```

Never overwrite a live `.env` during an update. Review changes to the tracked template between the
recorded old and new commits, then use `sudoedit .env` only for settings that must be adopted. Keep
the real file in the deployment's secret/configuration backup; do not print its values in a diff or
commit it.

At minimum, replace `PUBLIC_BASE_URL` and `MINIO_ROOT_PASSWORD`. Keep the API, Milvus, and ReID
listeners on loopback. Either configure application Basic Auth or make the TLS reverse proxy the
documented authentication boundary. The installer rejects the tracked placeholder secret and a
non-HTTPS public URL.

Seed these operator-managed model assets before running the wrapper:

```text
/var/lib/sightindex/models/yolo11n.pt
/var/lib/sightindex/models/insightface/models/buffalo_l/det_10g.onnx
/var/lib/sightindex/models/insightface/models/buffalo_l/w600k_r50.onnx
/var/lib/sightindex/models/sapiensid_wb12m/model.pth
/var/lib/sightindex/models/sapiensid_wb12m/model.yaml
/var/lib/sightindex/.cache/yolov8n-pose.pt
```

Make each file readable by `sightindex`. Review
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) before enabling SapiensID; its upstream code
and weights are subject to non-commercial terms.

For an update, take a filesystem or object-storage snapshot of `DATA_DIR` after all writers are
stopped, as shown above. After the source revision is selected, the wrapper confirms the services
remain inactive before changing runtime dependencies. It automatically backs up SQLite or the
repository-managed PostgreSQL container, builds dependencies as the unprivileged deployment
account, retains the previous virtual environment and frontend bundle, starts Milvus, then starts
ReID and the API in dependency order.

The optional `sightindex-embedding` service has a separate dependency profile and must be stopped
and disabled before using this wrapper. Manage that service as a separate reviewed deployment if
it is required.

```bash
cd /opt/sightindex
sudo bash deploy/rtx5090/install_or_update.sh ${BACKFILL_OPTION:-}
```

ReID model warmup can take up to five minutes. Useful options are shown by:

```bash
bash deploy/rtx5090/install_or_update.sh --help
```

`--skip-backup` is an explicit acknowledgement that a separate database backup has already been
verified. `--skip-deps` and `--skip-frontend` reuse artifacts only when every path is owned by
`sightindex-deploy` and none is writable by `sightindex`; otherwise rebuild them.

Run the verifier as the service identity so CUDA access, model readability, HOME, and `.env`
permissions match the real process:

```bash
sudo -u sightindex -H bash /opt/sightindex/deploy/rtx5090/verify.sh
```

It checks systemd state, a real CUDA tensor operation, ReID model identity and readiness, a
database-backed API, critical OpenAPI routes, Milvus write/search/delete, a synthetic InsightFace
ONNX session, and the built `/reid` console. It does not prove that cameras are reachable or that
new frames are advancing; verify that separately with a test stream and the external HTTPS URL.

The dependency files use compatible version ranges rather than a hardware lockfile. Record
`pip freeze`, the NVIDIA driver, Python, PyTorch, CUDA, ONNX Runtime, and model revisions for every
accepted host image before treating it as a repeatable production baseline.

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
until sudo docker compose exec -T postgres \
  sh -c 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'; do sleep 2; done
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
sudo -u sightindex -H sh -c 'cd /opt/sightindex && .venv/bin/python scripts/check_milvus.py'
```

`scripts/check_milvus.py` writes, searches, validates, and removes a temporary vector. The HTTP
health endpoint alone does not prove that the configured collection path works.

Keep `MILVUS_NAMESPACE_ID` stable when moving the same logical database between hosts. Use a new
visual collection prefix when the model or vector dimension changes.

## 6. Optional model services

Start optional dependencies before the API so readiness can be checked independently.

### Model inventory

The model assets each capability depends on are listed below. Except for entries marked as vendored
in the repository or operator-supplied images, every asset must be placed and hash-checked in
advance: requests never trigger automatic downloads. `FACE_INSIGHTFACE_ALLOW_DOWNLOAD` defaults to
`false`, and the ReID pose weights must also be pre-seeded before the service reports ready.

| Capability | Model / asset | Size / dims | Source and license |
| --- | --- | --- | --- |
| Person detection (in-API, `PERSON_DETECTOR=yolo`) | Ultralytics `yolo11n.pt` | ~6 MB | Official Ultralytics release, AGPL-3.0 |
| Pose keypoints (shared by appearance attributes and ReID alignment) | `yolov8n-pose.pt`, pre-seeded at `~/.cache/yolov8n-pose.pt` | 6.5 MB, 17 keypoints | Official Ultralytics release; participates in the immutable ReID pipeline revision, no automatic download |
| Face detection and embedding (in-API) | InsightFace `buffalo_l` ONNX set | 512-d output | InsightFace model zoo; note the upstream non-commercial restriction. When switching, set `FACE_EMBEDDING_DIM=512` and rebuild existing face vectors |
| ReID identity vector (optional service `18031`) | SapiensID `sapiensid_wb12m` (`model.yaml` + `model.pth`) | 1.5 GB, 4096-d | Upstream [mk-minchul/sapiensid](https://github.com/mk-minchul/sapiensid), CC BY-NC 4.0 non-commercial; not in Git — obtain it, then verify against the `/ready` fingerprint. The 28 MB DFA face-aligner weight is vendored in the repository |
| Generic visual embedding (optional service `18021`, container path `18032`) | `Qwen/Qwen3-VL-Embedding-2B` | 2B params, 2048-d | ModelScope; download and sha256-verify via `deploy/containers/download_embedding_model.py`. Alternatives: CLIP `sentence-transformers/clip-ViT-B-32` (512-d; generic image-text vectors, not used by the production business-search path) or the DashScope multimodal API |
| Text semantic embedding (semantic search, optional) | Ollama `qwen3-embedding:4b` | 4B params, 2560-d | Ollama model library; `EMBEDDING_DIM=2560` must match the model |
| VLM captioning / structured attributes (optional) | Any OpenAI-compatible endpoint, default `http://127.0.0.1:8001/v1` | depends on the chosen model | Operator-supplied (for example a vLLM-hosted Qwen-VL); `VLM_MODEL` names it |
| Qwen reranker helper (optional `18022`) | Operator-supplied NVIDIA image + `VLM_RERANK_MODEL` | — | The repository does not build this image |
| External YOLO service (optional `19121`) | Operator-supplied image + `YOLO_SERVICE_MODEL_PATH` | — | The repository does not build this image |

Read-only model directory layout for the container deployment (`MODEL_DIR`):

```text
models/
  yolo11n.pt
  yolov8n-pose.pt
  sapiensid_wb12m/model.yaml
  sapiensid_wb12m/model.pth
  insightface/models/buffalo_l/*.onnx
  qwen3-vl-embedding-2b-c73fa9ca/
```

Model and third-party usage restrictions are also covered in `THIRD_PARTY_NOTICES.md` and
[`deploy/agx/reid_service/README.md`](../deploy/agx/reid_service/README.md).

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
curl --fail --user operator \
  http://127.0.0.1:8000/api/media/counts
```

With only the username supplied, curl prompts for the password instead of putting it in shell
history or the process command line. If application Basic Auth is disabled behind a trusted local
proxy, omit `--user`.

For ReID, a `200` from the API status route is not enough; inspect the JSON fields:

```bash
curl --fail --user operator \
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
5. Install the reviewed dependency set and rebuild `frontend/dist`.
6. Validate configuration and run tests.
7. Start dependencies, model services, then the API.
8. Run liveness, database, vector, and external smoke checks.

Example database backup:

```bash
set -euo pipefail
cd /opt/sightindex
PREVIOUS_COMMIT=$(sudo -u sightindex-deploy -H git -C /opt/sightindex rev-parse HEAD)
BACKUP_DIR=/var/lib/sightindex/backups
DATABASE_BACKUP="$BACKUP_DIR/sightindex-${PREVIOUS_COMMIT}.dump"
DATA_BACKUP="$BACKUP_DIR/data-${PREVIOUS_COMMIT}.tar.gz"
DATABASE_PARTIAL="${DATABASE_BACKUP}.partial"
DATA_PARTIAL="${DATA_BACKUP}.partial"
sudo systemctl stop sightindex-api sightindex-reid sightindex-embedding \
  sightindex-attribute-backfill 2>/dev/null || true
for unit in sightindex-api sightindex-reid sightindex-embedding sightindex-attribute-backfill; do
  if sudo systemctl is-active --quiet "$unit"; then
    echo "failed to stop $unit" >&2
    exit 1
  fi
done
sudo install -d -m 700 -o root -g root "$BACKUP_DIR"
sudo install -m 600 /dev/null "$DATABASE_PARTIAL"
sudo install -m 600 /dev/null "$DATA_PARTIAL"
sudo sh -c 'cd /opt/sightindex && docker compose --env-file .env exec -T postgres \
  sh -c '\''pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc'\'' > "$1"' \
  sh "$DATABASE_PARTIAL"
sudo test -s "$DATABASE_PARTIAL"
sudo sh -c 'cd /opt/sightindex && docker compose --env-file .env exec -T postgres \
  pg_restore --list < "$1" >/dev/null' sh "$DATABASE_PARTIAL"
sudo mv "$DATABASE_PARTIAL" "$DATABASE_BACKUP"
sudo tar -C /var/lib/sightindex -czf "$DATA_PARTIAL" data
sudo tar -tzf "$DATA_PARTIAL" >/dev/null
sudo mv "$DATA_PARTIAL" "$DATA_BACKUP"
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
- Restrict `.env`, database backups, media directories, and model caches to their documented
  service, deployment, or root owner.
- Do not log tokens, full RTSP URLs, or raw biometric payloads.
- Review third-party model licenses before commercial or biometric use.

## Operational updates (2026-09)

### Code sync and release checks

Sync production code with `deploy/rtx5090/sync_code.sh <user@host> <target-dir>`. The script
explicitly excludes `.env`, `data/`, the root `sightindex.db` plus all SQLite sidecar files,
model weights, and logs. Do not replace it with a plain `rsync`, and never run `rsync --delete`
against the project root: that deletes models and field data that are not in Git. The installer
takes an online SQLite backup into `backups/` on every run, restarts the ReID and API services,
and waits up to 300 seconds for model warm-up.

Before releasing, run locally:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check app tests scripts
cd frontend
npm run build
npm run test:reid
```

`test:reid` drives the Vue compiler and reactivity runtime to verify query switching, stale
responses, manual labeling, camera grouping, lightbox focus, and evidence copy. It touches no
production service or real imagery and does not replace visual acceptance in a browser. Both
deployment entry points run this regression while building the frontend; with `--skip-frontend`
you must have passed the build and regression locally first.

When the server was synced without `.git`, verify the release by comparing the recorded release
commit and SHA-256 of key code and frontend build files; a failing `git` command is not evidence
that versions match.

### ReID calibration feedback table

The first API start after release creates the `reid_match_feedback` table through the existing
`init_db()`. Manual confirmations on the console only write to this calibration table; they never
change `person_crops.person_id` or live ranking. Export labels at `/api/reid/feedback/export.csv`
and evaluate thresholds per `docs/reid-walkthrough-calibration.md`.

### ReID face-safety migration and acceptance

The update adds two nullable columns to `crop_face_extractions`: `absence_reason` and
`input_fingerprint`. With `AUTO_CREATE_TABLES=true` the compatible migration runs automatically at
API startup, is idempotent, and preserves existing rows. `verify.sh` now checks both columns and
the `face_coverage` field in search and cross-camera responses; an HTTP 200 alone does not
complete acceptance. If automatic table creation is disabled, back up the database, run
`.venv/bin/python -c 'from app.db.session import init_db; init_db()'` in a maintenance window,
then start the new API. Multi-worker deployments must finish the migration before starting
workers rather than initializing concurrently.

- `FACE_NEGATIVE_CACHE_TTL_SECONDS=300` only controls the short-lived cache of deterministic
  abstentions; it is not a face-quality threshold.
- Legacy reason-less empty cache entries retry lazily per query; temporary file, read, or
  inference failures add no negative cache entries; changes to the original image or person box
  invalidate the negative cache immediately. No Milvus wipe or full body-index rebuild is needed.
  Early queries may be slower while faces re-extract; replay a small batch first and watch
  latency.
- Valid positive cache entries are reused; new entries record the source-image fingerprint.
  Upscaling no longer changes the recognition model name, and legacy upscale suffixes stay
  readable.
- Crops without body vectors stay as independent frames instead of guessing identity by time;
  appearance counts may rise, which is not an index-coverage regression.
- At most two candidate frames are consulted per appearance. Borrowed faces only influence soft
  ranking and can neither hard-confirm nor hard-exclude; cards disclose the supplemental frame
  source.
- Acceptance also confirms `REID_FACE_RESCUE_MIN_BODY_SCORE` does not exceed the normal body
  candidate threshold: it only widens the face-review pool and never lowers the formal body
  admission line; low-score candidates without a reliable face match are still dropped.

When rolling back application code, the two new nullable columns can stay: old code ignores them,
and neither dropping tables nor restoring the whole database over new observation data is
required. If a database restore is unavoidable, stop writers first and assess the data loss after
the backup point; a code rollback is not authorization for a database rollback. Real-world
accuracy still requires a manually labeled multi-person cross-camera walkthrough; synthetic
regressions and successful model loads do not substitute.

Face review uses the target person box within the original frame and does not treat bystander
faces in padded display crops as the target. It abstains on multiple or unlinkable faces; native
face pixels, sharpness, and five-point symmetry jointly cap the quality score, and upscaling does
not raise native quality. The `identity-v2` cache signature invalidates older extraction results
automatically, and no-face results are cached durably. These quality parameters are conservative
defaults pending walkthrough calibration.

### Stream heartbeat and decoder subprocess

Check capture health from `/api/streams` twice, roughly 10 seconds apart, using only the
read-frame heartbeat — not stored image counts or `updated_at`:

- `last_frame_read_at` (UTC), `consecutive_read_failures`, and `capture_health` are per stream.
- A successful frame read refreshes the heartbeat even when nobody is in view, the stream is
  warming up, or frames are skipped by quality gates (writes are throttled to 5 seconds and also
  follow the sampling interval).
- `healthy` means the heartbeat is fresh; `stalled` means no update for more than
  `max(30s, 3 × sampling interval)`; `unverified` means no frame yet or reconnecting.
- `updated_at` changes on other writes such as status edits and never proves RTSP reads on its
  own.

RTSP decoding now runs in a separate subprocess. Even when a native `read()` ignores its timeout,
the whole decoder process is terminated, reaped, and reconnected, so permanently blocked threads
no longer accumulate. Normal API upgrades reap subprocesses and preserve the auto-start intent of
running cameras; streams stopped by a user are not restarted.

### Background VLM structured attributes

Recommended flags for the RTX 5090 host: `VLM_STRUCTURED_BACKGROUND=true` and
`VLM_STRUCTURED_ON_INGEST=false`. New crops and `person_attributes` jobs persist in the database
and a dedicated consumer thread runs the VLM without blocking RTSP capture or the ReID index
consumer. A sweep every minute backfills historical unlabeled crops; failures retry with queue
backoff, expired leases recover after process restarts, and jobs that reach the retry limit stay
`failed` instead of resetting forever. When the queue is full, captured images are still saved
and enqueued later by a database scan.

```bash
# Current coverage and failure records (not yesterday's backfill completion marker)
curl -fsS http://127.0.0.1:18030/api/attributes/jobs
# After fixing the cause, retry one crop; completed labels are not overwritten.
curl -fsS -X POST http://127.0.0.1:18030/api/attributes/jobs/CROP_UUID/retry
```

`pending_crops` includes unfinished and quarantined-failed crops; coverage is complete only when
both it and `queue` are empty. The old `sightindex-attribute-backfill.service` remains a manual
batch tool; continuous processing of new data belongs to the persistent job queue inside the API.

### Manual acceptance for the ReID console

HTTP 200 only proves the page is reachable. Also confirm:

- The query image matches the current candidates; after switching from a `crop_id` page to an
  uploaded image, stale cross-camera leads and labeling buttons must not linger.
- Cards keep image, time, body similarity, and a one-line conclusion on the first layer; labels,
  faces, and provenance appear only under expanded evidence details.
- "Cross-camera leads" never implies a confirmed visit; only human feedback shows as manually
  confirmed, and algorithm scores are not same-person probabilities.
- Cameras carrying a reliable face keep their priority; frontend numeric re-sorting must not
  override it, and results within one camera preserve API order.
- Candidate images render at equal height without nested scroll boxes on narrow screens; every
  main image zooms, Tab stays inside the preview, and Esc returns focus to the source button.
- Empty face evidence never renders as "inconsistent", and fewer than two high-confidence labels
  shows neutral instead of fabricating 100% agreement.

The adopted scope, rejected suggestions, and residual risks of this review round are recorded in
`docs/reid-fable5-review-20260905.md`.
