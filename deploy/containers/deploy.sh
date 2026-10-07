#!/usr/bin/env bash
# Deploy a reviewed container release, preserving models, media and persistent volumes.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROOT="${SIGHTINDEX_ROOT:-/data/sightindex-bj-test}"
ENV_FILE=""
RELEASE=""
STACKS="base"
PROJECT_NAME=""
DO_BUILD=1
CHECK_ONLY=0
SKIP_BACKUP=0
ALLOW_DIRTY=0
MODEL_MANIFEST=""
MODEL_SOURCE=""
PREPARE_MODELS=0
DOWNLOAD_MODELS=0
ACK_MODEL_TERMS=0
BUILD_MODEL_SERVICES=0
LOCKED=0
LIVE_CHANGE=0
BACKUP=""
PYTHON="${SIGHTINDEX_DEPLOY_PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage: bash deploy.sh [--target containers] [options]
  --root DIR          Deployment root (default /data/sightindex-bj-test)
  --source DIR        Reviewed source checkout (default this script's checkout)
  --env-file FILE     Private configuration (default ROOT/.env)
  --release NAME      Unique release name (default timestamp + Git revision)
  --stacks "base [reid] [embedding] [semantic]"
  --project-name NAME  Independent Compose project (default sightindex-bj-test)
  --check             Read-only config/model/tool preflight; creates nothing
  --no-build          Use the existing local SIGHTINDEX_IMAGE (still verify it)
  --allow-dirty       Explicitly permit uncommitted code; label it dirty
  --skip-backup       Only with a separately verified external database backup
  --model-manifest FILE  Reviewed model lockfile; check SHA-256 before deployment
  --prepare-models    Prepare model files before the application preflight
  --model-source DIR  Import offline model files from the reviewed bundle
  --download-models  Explicitly fetch missing pinned assets (never implicit)
  --acknowledge-model-terms  Record review; does not grant commercial model rights
  --build-model-services  Build the selected Qwen embedding image from the app image
  --set-image         Compatibility flag; new builds always select their new image
  -h, --help          Show help

Review private configuration and model sources first. Model download requires
explicit options and reviewed metadata. This script does not change drivers,
start camera streams or restore databases.
EOF
}
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --root|--source|--env-file|--release|--stacks|--project-name|--model-manifest|--model-source)
      [ "$#" -ge 2 ] || fail "missing value for $1"
      case "$1" in
        --root) ROOT="$2" ;; --source) SOURCE="$2" ;; --env-file) ENV_FILE="$2" ;;
        --release) RELEASE="$2" ;; --stacks) STACKS="$2" ;;
        --project-name) PROJECT_NAME="$2" ;;
        --model-manifest) MODEL_MANIFEST="$2" ;; --model-source) MODEL_SOURCE="$2" ;;
      esac
      shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    --no-build) DO_BUILD=0; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --skip-backup) SKIP_BACKUP=1; shift ;;
    --prepare-models) PREPARE_MODELS=1; shift ;;
    --download-models) DOWNLOAD_MODELS=1; PREPARE_MODELS=1; shift ;;
    --acknowledge-model-terms) ACK_MODEL_TERMS=1; shift ;;
    --build-model-services) BUILD_MODEL_SERVICES=1; shift ;;
    --set-image) shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument; use --help" ;;
  esac
done

# Validate targets before mkdir, configuration generation, builds or service changes.
[[ "$ROOT" =~ ^/[A-Za-z0-9_./-]+$ ]] || fail "--root must be an absolute path without spaces or shell syntax"
case "$ROOT" in *//*|*/../*|*/./*|*/..|*/.) fail "deployment root must be canonical, without //, . or .. segments" ;; esac
ROOT="${ROOT%/}"
case "$ROOT" in ''|/|/tmp|/var|/opt|/data|/Users|/etc|/usr|/bin|/sbin|/Library|/Applications|/private|"${HOME:-/nonexistent}") fail "use a dedicated deployment root" ;; esac
[ ! -L "$ROOT" ] || fail "deployment root must not be a symlink"
[ -d "$SOURCE" ] || fail "source directory missing"
SOURCE="$(cd "$SOURCE" && pwd -P)"
[ "$ROOT" != "$SOURCE" ] || fail "deployment root must be separate from source"
case "$ROOT/" in "$SOURCE/"*) fail "deployment root must not be inside the source checkout" ;; esac
PREFLIGHT="$SOURCE/deploy/containers/preflight.py"
[ -f "$PREFLIGHT" ] && [ -f "$SOURCE/deploy/containers/manage.sh" ] || fail "source lacks deployment tools"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
case "$ENV_FILE" in /*) ;; *) fail "--env-file must be absolute" ;; esac
[ ! -L "$ENV_FILE" ] || fail "env file must not be a symlink"
if [ -n "$PROJECT_NAME" ]; then
  [[ "$PROJECT_NAME" =~ ^[a-z0-9][a-z0-9_-]{0,62}$ ]] || fail "invalid Compose project name"
fi
read -r -a STACK_ARRAY <<< "$STACKS"
[ "${#STACK_ARRAY[@]}" -gt 0 ] || fail "select at least base stack"
has_base=0
for stack in "${STACK_ARRAY[@]}"; do
  case "$stack" in base) has_base=1 ;; reid|embedding|semantic) ;; *) fail "unknown stack; use base/reid/embedding/semantic" ;; esac
done
[ "$has_base" = 1 ] || fail "stacks must include base"
[ -z "$MODEL_SOURCE" ] || [ "$DOWNLOAD_MODELS" = 0 ] || fail "choose offline model source or download, not both"
if [ "$PREPARE_MODELS" = 1 ]; then
  [ -n "$MODEL_MANIFEST" ] || fail "model preparation requires --model-manifest"
  if [ "$CHECK_ONLY" = 0 ]; then
    [ "$ACK_MODEL_TERMS" = 1 ] || fail "model preparation requires --acknowledge-model-terms (not a commercial license grant)"
    [ -n "$MODEL_SOURCE" ] || [ "$DOWNLOAD_MODELS" = 1 ] || fail "select --model-source or --download-models"
  fi
elif [ -n "$MODEL_SOURCE" ] || [ "$ACK_MODEL_TERMS" = 1 ]; then
  fail "model source/terms options require --prepare-models"
fi
if [ "$BUILD_MODEL_SERVICES" = 1 ]; then
  case " $STACKS " in *" embedding "*) ;; *) fail "--build-model-services requires embedding stack" ;; esac
fi
timeout_seconds="${SIGHTINDEX_DEPLOY_TIMEOUT_SECONDS:-600}"
[[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] || fail "deployment timeout must be a positive integer"
# Resolve configuration only after every authority/argument check above.
# The selected root is bound to its accepted project. Never retarget an existing
# root or let ambient COMPOSE_PROJECT_NAME silently select another deployment.
configured_project=""
if [ -f "$ENV_FILE" ]; then
  configured_project="$("$PYTHON" "$PREFLIGHT" --env-file "$ENV_FILE" --value COMPOSE_PROJECT_NAME)"
fi
accepted_project=""
if [ -e "$ROOT/project-name" ] || [ -L "$ROOT/project-name" ]; then
  [ -f "$ROOT/project-name" ] && [ ! -L "$ROOT/project-name" ] || fail "invalid project-name marker"
  accepted_project="$(<"$ROOT/project-name")"
  [[ "$accepted_project" =~ ^[a-z0-9][a-z0-9_-]{0,62}$ ]] || fail "invalid accepted project name"
fi
PROJECT_NAME="${PROJECT_NAME:-${configured_project:-${accepted_project:-sightindex-bj-test}}}"
[[ "$PROJECT_NAME" =~ ^[a-z0-9][a-z0-9_-]{0,62}$ ]] || fail "invalid Compose project name"
[ -z "$configured_project" ] || [ "$PROJECT_NAME" = "$configured_project" ] || fail "project name differs from private configuration"
[ -z "$accepted_project" ] || [ "$PROJECT_NAME" = "$accepted_project" ] || fail "deployment root belongs to a different project"
if [ "$CHECK_ONLY" = 1 ]; then
  if [ -n "$MODEL_MANIFEST" ]; then
    "$PYTHON" "$SOURCE/deploy/models/setup.py" --target containers --source "$SOURCE" \
      --env-file "$ENV_FILE" --stacks "$STACKS" --manifest "$MODEL_MANIFEST" --check
  fi
  exec "$PYTHON" "$PREFLIGHT" --target containers --source "$SOURCE" \
    --env-file "$ENV_FILE" --stacks "$STACKS" --check-tools
fi
revision="unknown"
if git -C "$SOURCE" rev-parse HEAD >/dev/null 2>&1; then
  revision="$(git -C "$SOURCE" rev-parse HEAD)"
  dirty="$(git -C "$SOURCE" status --porcelain --untracked-files=normal)"
  if [ -n "$dirty" ]; then
    [ "$ALLOW_DIRTY" = 1 ] || fail "source is dirty; commit/review it or explicitly pass --allow-dirty"
    revision="${revision}-dirty"
  fi
else
  [ "$ALLOW_DIRTY" = 1 ] || fail "source has no Git revision; use a reviewed checkout or explicitly --allow-dirty"
  revision="unknown-dirty"
fi
RELEASE="${RELEASE:-$(date +%Y%m%d-%H%M%S)-${revision:0:12}}"
[[ "$RELEASE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "release name must contain only letters, digits, . _ -"
[ ! -e "$ROOT/releases/$RELEASE" ] || fail "release already exists"
command -v "$PYTHON" >/dev/null 2>&1 || fail "Python 3.11+ is required"
command -v docker >/dev/null 2>&1 || fail "Docker is required"
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required"
cleanup() {
  local code="$?"
  if [ "$code" -ne 0 ]; then
    printf 'Deployment failed; this release was NOT accepted.\n' >&2
    if [ "$LIVE_CHANGE" = 1 ]; then
      printf 'Services may be on the candidate version. Inspect logs and use the explicit rollback procedure; no database was restored.\n' >&2
    fi
    [ -z "$BACKUP" ] || printf 'Private configuration/database backup: %s\n' "$BACKUP" >&2
  fi
  [ "$LOCKED" = 0 ] || rmdir "$ROOT/.deploy-lock" 2>/dev/null || true
  exit "$code"
}
trap cleanup EXIT
mkdir -p "$ROOT"
mkdir "$ROOT/.deploy-lock" 2>/dev/null || fail "deployment lock exists; confirm no deployment is running before removing it"
LOCKED=1
for directory in releases backups media models cache cache/api; do
  [ ! -L "$ROOT/$directory" ] || fail "deployment directory must not be a symlink: $directory"
  if [ ! -d "$ROOT/$directory" ]; then
    mkdir -p "$ROOT/$directory"
    case "$directory" in
      media|cache|cache/api)
        if [ "$(id -u)" != 1000 ]; then
          chown 1000:1000 "$ROOT/$directory" 2>/dev/null \
            || fail "new $directory must be writable by container UID 1000; prepare ownership manually"
        fi ;;
      models) chmod 755 "$ROOT/$directory" ;;
    esac
  fi
done
chmod 700 "$ROOT/backups"
env_value() { "$PYTHON" "$PREFLIGHT" --env-file "$ENV_FILE" --value "$1"; }
set_env_value() {
  local key="$1" value="$2" temporary
  temporary="$(mktemp "$(dirname "$ENV_FILE")/.sightindex-env.XXXXXX")"
  awk -F= -v key="$key" -v value="$value" '
    $1 == key { print key "=" value; found=1; next }
    { print } END { if (!found) print key "=" value }
  ' "$ENV_FILE" > "$temporary"
  chmod 600 "$temporary"
  mv "$temporary" "$ENV_FILE"
}
random_hex() { "$PYTHON" -c 'import secrets; print(secrets.token_hex(24))'; }
if [ ! -e "$ENV_FILE" ]; then
  [ -d "$(dirname "$ENV_FILE")" ] || fail "env parent directory must exist"
  sed -e "s#^MEDIA_DIR=.*#MEDIA_DIR=$ROOT/media#" \
      -e "s#^MODEL_DIR=.*#MODEL_DIR=$ROOT/models#" \
      -e "s#^CACHE_DIR=.*#CACHE_DIR=$ROOT/cache#" \
      "$SOURCE/deploy/containers/.env.example" > "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  for key in APP_BASIC_AUTH_PASSWORD POSTGRES_PASSWORD MINIO_ROOT_PASSWORD REID_SERVICE_API_KEY; do
    set_env_value "$key" "$(random_hex)"
  done
  set_env_value COMPOSE_PROJECT_NAME "$PROJECT_NAME"
  printf 'Generated private configuration: %s. Prepare models and review it before retrying.\n' "$ENV_FILE"
fi
if [ -n "$MODEL_MANIFEST" ]; then
  model_args=(--target containers --source "$SOURCE" --env-file "$ENV_FILE" --stacks "$STACKS" --manifest "$MODEL_MANIFEST")
  if [ "$PREPARE_MODELS" = 1 ]; then
    model_args+=(--acknowledge-model-terms)
    if [ "$DOWNLOAD_MODELS" = 1 ]; then model_args+=(--download); else model_args+=(--source-dir "$MODEL_SOURCE"); fi
  else
    model_args+=(--check)
  fi
  "$PYTHON" "$SOURCE/deploy/models/setup.py" "${model_args[@]}"
fi
"$PYTHON" "$PREFLIGHT" --target containers --source "$SOURCE" \
  --env-file "$ENV_FILE" --stacks "$STACKS" --check-tools

# Compose must not auto-create root-owned writable bind directories. Only set
# ownership on directories this invocation creates, never recurse into user data.
media_directory="$(env_value MEDIA_DIR)"
cache_directory="$(env_value CACHE_DIR)"
writable_directories=("$media_directory" "$cache_directory" "$cache_directory/api")
case " $STACKS " in *" embedding "*) writable_directories+=("$cache_directory/embedding") ;; esac
for directory in "${writable_directories[@]}"; do
  if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
    if [ "$(id -u)" != 1000 ]; then
      chown 1000:1000 "$directory" 2>/dev/null \
        || fail "new writable bind directory needs UID 1000 ownership; prepare it manually"
    fi
  fi
done

PREVIOUS=""
if [ -f "$ROOT/active-release" ]; then
  [ ! -L "$ROOT/active-release" ] || fail "active release marker must not be a symlink"
  PREVIOUS="$(<"$ROOT/active-release")"
else
  for candidate in "$ROOT"/releases/*; do
    if [ -f "$candidate/deploy/containers/compose.yaml" ] && [ ! -f "$candidate/.deployment-pending" ]; then
      PREVIOUS="$candidate"
    fi
  done
fi
if [ -n "$PREVIOUS" ]; then
  [ -d "$PREVIOUS" ] || fail "previous release directory is missing"
  PREVIOUS="$(cd "$PREVIOUS" && pwd -P)"
  case "$PREVIOUS" in "$ROOT"/releases/*) ;; *) fail "previous release escapes deployment root" ;; esac
fi
BACKUP="$ROOT/backups/$RELEASE"
mkdir -m 700 "$BACKUP"
cp "$ENV_FILE" "$BACKUP/config.env"
chmod 600 "$BACKUP/config.env"
printf '%s\n' "$PREVIOUS" > "$BACKUP/previous-release"
[ ! -f "$ROOT/active-stacks" ] || cp "$ROOT/active-stacks" "$BACKUP/previous-stacks"
if [ "$SKIP_BACKUP" = 1 ]; then
  printf 'WARNING: database backup explicitly skipped; external verified backup is required.\n' >&2
elif [ -n "$PREVIOUS" ]; then
  PREVIOUS_COMPOSE=(docker compose --project-name "$PROJECT_NAME" --env-file "$ENV_FILE" -f "$PREVIOUS/deploy/containers/compose.yaml")
  postgres_id="$("${PREVIOUS_COMPOSE[@]}" ps --all --quiet postgres)"
  if [ -n "$postgres_id" ]; then
    if ! "${PREVIOUS_COMPOSE[@]}" exec -T postgres pg_dump -U sightindex -d sightindex -Fc \
      > "$BACKUP/postgres.dump.partial" 2> "$BACKUP/backup-error.log"; then
      fail "database backup failed; inspect private backup-error.log before retrying"
    fi
    [ -s "$BACKUP/postgres.dump.partial" ] || fail "database backup is empty"
    mv "$BACKUP/postgres.dump.partial" "$BACKUP/postgres.dump"
  elif docker volume inspect "${PROJECT_NAME}_postgres-data" >/dev/null 2>&1; then
    fail "existing database volume has no backup-capable postgres container; start the prior database or use a verified external backup"
  fi
elif docker volume inspect "${PROJECT_NAME}_postgres-data" >/dev/null 2>&1; then
  fail "existing database volume but no prior release; supply a verified external backup and --skip-backup"
fi

RELEASE_DIR="$ROOT/releases/$RELEASE"
mkdir "$RELEASE_DIR"
: > "$RELEASE_DIR/.deployment-pending"
# Explicit source allowlist: private env, databases, media, model weights and
# unrelated operator reports never enter the release archive or build context.
assets=(app main.py pyproject.toml frontend deploy .dockerignore)
for optional in requirements.txt requirements-dev.txt THIRD_PARTY_NOTICES.md LICENSE README.md README.zh-CN.md deploy.sh; do
  [ ! -f "$SOURCE/$optional" ] || assets+=("$optional")
done
# Keep only the documented env template in the snapshot. This manifest avoids
# copying symlink targets, source secrets, runtime media or binary checkpoints.
( cd "$SOURCE" && "$PYTHON" - "${assets[@]}" <<'PY' | tar -cf - -T -
from pathlib import Path
import sys

blocked_directories = {"node_modules", "dist", ".venv", "__pycache__", "data", "cache", "models", "weights", "backups", ".pytest_cache", ".ruff_cache", ".git", ".sightindex-model-assets-quarantine"}
blocked_suffixes = {".pt", ".pth", ".onnx", ".db", ".sqlite", ".sqlite3", ".safetensors", ".bin", ".pyc", ".mp4", ".avi", ".webm", ".mov", ".png", ".jpg", ".jpeg"}
for root in sys.argv[1:]:
    path = Path(root)
    candidates = [path] if path.is_file() else sorted(path.rglob("*"))
    for item in candidates:
        if item.is_symlink() or not item.is_file():
            continue
        if any(part in blocked_directories and not (part == "models" and item.parts[:2] in {("app", "models"), ("deploy", "models")}) for part in item.parts):
            continue
        is_frontend_asset = str(item).startswith(("frontend/src/assets/", "frontend/public/")) and item.suffix.lower() in {".png", ".jpg", ".jpeg"}
        if (any(suffix.lower() in blocked_suffixes for suffix in item.suffixes) and not is_frontend_asset) or "\n" in str(item):
            continue
        if item.name.endswith(".partial") or item.name == ".sightindex-model-assets.lock":
            continue
        if (item.name.startswith(".env") or item.suffix == ".env") and item.name != ".env.example":
            continue
        if item.name.endswith(tuple(extension + sidecar for extension in (".db", ".sqlite", ".sqlite3") for sidecar in ("-wal", "-shm", "-journal"))) or ".safetensors" in item.name:
            continue
        print(item)
PY
    ) \
  | ( cd "$RELEASE_DIR" && tar -xf - )
printf '%s\n' "$revision" > "$RELEASE_DIR/SOURCE_REVISION"
"$PYTHON" - "$RELEASE_DIR" "$revision" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = {}
for path in sorted(root.rglob("*")):
    if path.is_file() and not path.is_symlink() and not path.name.startswith(".deployment-"):
        files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {"source_revision": sys.argv[2], "files_sha256": files}
(root / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
install -m 0755 "$RELEASE_DIR/deploy/containers/manage.sh" "$ROOT/manage.sh"
IMAGE_TAG="sightindex:$RELEASE"
if [ "$PROJECT_NAME" != sightindex-bj-test ]; then
  IMAGE_TAG="$PROJECT_NAME:$RELEASE"
fi
if [ "$DO_BUILD" = 1 ]; then
  docker build -f "$RELEASE_DIR/deploy/containers/Dockerfile" \
    --build-arg SOURCE_REVISION="$revision" -t "$IMAGE_TAG" "$RELEASE_DIR"
  set_env_value SIGHTINDEX_IMAGE "$IMAGE_TAG"
else
  IMAGE_TAG="$(env_value SIGHTINDEX_IMAGE)"
  [ -n "$IMAGE_TAG" ] || fail "--no-build requires SIGHTINDEX_IMAGE"
  docker image inspect "$IMAGE_TAG" >/dev/null 2>&1 || fail "configured application image is not local"
fi
if [ "$BUILD_MODEL_SERVICES" = 1 ]; then
  embedding_image="$("$PYTHON" "$PREFLIGHT" --env-file "$ENV_FILE" --value QWEN_EMBEDDING_IMAGE)"
  [ -n "$embedding_image" ] && [ "$embedding_image" != "$IMAGE_TAG" ] || fail "Qwen embedding image must have a separate configured tag"
  docker build -f "$RELEASE_DIR/deploy/containers/Dockerfile.embedding" \
    --build-arg SIGHTINDEX_BASE_IMAGE="$IMAGE_TAG" -t "$embedding_image" "$RELEASE_DIR"
fi
docker image inspect --format '{{.Id}}' "$IMAGE_TAG" > "$RELEASE_DIR/IMAGE_ID"
SIGHTINDEX_ROOT="$ROOT" bash "$ROOT/manage.sh" --project-name "$PROJECT_NAME" --env-file "$ENV_FILE" --release "$RELEASE_DIR" config "${STACK_ARRAY[@]}"
LIVE_CHANGE=1
SIGHTINDEX_ROOT="$ROOT" bash "$ROOT/manage.sh" --project-name "$PROJECT_NAME" --env-file "$ENV_FILE" --release "$RELEASE_DIR" up "${STACK_ARRAY[@]}"
COMPOSE=(docker compose --project-name "$PROJECT_NAME" --env-file "$ENV_FILE" -f "$RELEASE_DIR/deploy/containers/compose.yaml")
for stack in "${STACK_ARRAY[@]}"; do
  case "$stack" in
    reid) COMPOSE+=(--profile reid) ;;
    embedding) COMPOSE+=(-f "$RELEASE_DIR/deploy/containers/compose.embedding.yaml" --profile embedding) ;;
    semantic) COMPOSE+=(-f "$RELEASE_DIR/deploy/containers/compose.semantic-search.yaml") ;;
  esac
done
deadline=$((SECONDS + timeout_seconds))
services=(postgres etcd minio milvus api)
for stack in "${STACK_ARRAY[@]}"; do
  case "$stack" in reid|embedding) services+=("$stack") ;; esac
done
printf 'Waiting for every selected service to be healthy (timeout %ss).\n' "$timeout_seconds"
while :; do
  ready=1
  for service in "${services[@]}"; do
    container_id="$("${COMPOSE[@]}" ps --quiet "$service")"
    [ -n "$container_id" ] || { ready=0; continue; }
    state="$(docker inspect --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}missing-healthcheck{{end}}' "$container_id")"
    [ "$state" = 'running healthy' ] || ready=0
  done
  [ "$ready" = 0 ] || break
  [ "$SECONDS" -lt "$deadline" ] || fail "selected services did not become healthy; inspect manage.sh logs api"
  sleep 3
done
SIGHTINDEX_ROOT="$ROOT" bash "$ROOT/manage.sh" --project-name "$PROJECT_NAME" --env-file "$ENV_FILE" --release "$RELEASE_DIR" \
  verify "${STACK_ARRAY[@]}" --model-smoke
# Promote only after image identity, contract, DB, model and isolated upload checks.
release_marker="$(mktemp "$ROOT/.active-release.XXXXXX")"
stacks_marker="$(mktemp "$ROOT/.active-stacks.XXXXXX")"
project_marker="$(mktemp "$ROOT/.project-name.XXXXXX")"
printf '%s\n' "$RELEASE_DIR" > "$release_marker"
printf '%s\n' "$STACKS" > "$stacks_marker"
printf '%s\n' "$PROJECT_NAME" > "$project_marker"
mv "$project_marker" "$ROOT/project-name"
mv "$release_marker" "$ROOT/active-release"
mv "$stacks_marker" "$ROOT/active-stacks"
rm "$RELEASE_DIR/.deployment-pending"
printf 'Accepted release: %s\nImage: %s\nPrivate backup: %s\n' "$RELEASE_DIR" "$IMAGE_TAG" "$BACKUP"
printf 'Only the selected stacks passed startup acceptance. Accuracy, historical coverage and camera recording need separate acceptance.\n'
