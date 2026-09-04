#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPECTED_ROOT="/opt/sightindex"
SERVICE_USER="sightindex"
SERVICE_GROUP="sightindex"
SERVICE_HOME="/var/lib/sightindex"
DEPLOY_USER="sightindex-deploy"
DEPLOY_HOME="/var/lib/sightindex-deploy"
ENV_TEMPLATE="$ROOT_DIR/deploy/rtx5090/sightindex.env.example"
ENV_FILE="$ROOT_DIR/.env"
LOG_DIR="/var/log/sightindex"
BACKUP_DIR="$SERVICE_HOME/backups"
LOCK_FILE="/run/lock/sightindex-deploy.lock"
SKIP_DEPS=0
SKIP_FRONTEND=0
SKIP_MILVUS=0
SKIP_BACKUP=0
START_BACKFILL=0
SKIP_VERIFY=0
SERVICES_STOPPED=0
BACKFILL_WAS_ACTIVE=0
VENV_REPLACEMENT_STARTED=0
FRONTEND_REPLACEMENT_STARTED=0

usage() {
  cat <<'EOF'
Usage: sudo bash deploy/rtx5090/install_or_update.sh [options]

Install or update the reviewed checkout on an RTX 5090 host. This script never runs git pull;
select and review the release revision before invoking it.

Options:
  --skip-deps       Reuse the existing Python virtual environment.
  --skip-frontend   Reuse the existing frontend/dist bundle.
  --skip-milvus     Do not start or update the local Milvus Compose stack.
  --skip-backup     Continue without an automatic database backup.
  --start-backfill  Start or resume the structured-attribute backfill worker.
  --skip-verify     Skip the final deployment verification.
  -h, --help        Show this message.

Use --skip-backup only after an external database backup has been verified. SQLite and the
repository-managed PostgreSQL service are backed up automatically when present.
EOF
}

log() {
  printf '[sightindex-deploy] %s\n' "$*"
}

failure_hint() {
  if [ "$SERVICES_STOPPED" -eq 1 ]; then
    printf '%s\n' \
      '[sightindex-deploy] Services may remain stopped. Inspect the deploy logs and system journal before restarting.' \
      >&2
  fi
}

restore_previous_venv() {
  if [ "$VENV_REPLACEMENT_STARTED" -ne 1 ]; then
    return
  fi
  rm -rf "$ROOT_DIR/.venv"
  if [ -d "$ROOT_DIR/.venv.rollback" ]; then
    mv "$ROOT_DIR/.venv.rollback" "$ROOT_DIR/.venv"
    printf '%s\n' '[sightindex-deploy] Restored the previous virtual environment.' >&2
  fi
  VENV_REPLACEMENT_STARTED=0
}

restore_previous_frontend() {
  if [ "$FRONTEND_REPLACEMENT_STARTED" -ne 1 ]; then
    return
  fi
  rm -rf "$ROOT_DIR/frontend/dist"
  if [ -d "$ROOT_DIR/frontend/dist.rollback" ]; then
    mv "$ROOT_DIR/frontend/dist.rollback" "$ROOT_DIR/frontend/dist"
    printf '%s\n' '[sightindex-deploy] Restored the previous frontend bundle.' >&2
  fi
  FRONTEND_REPLACEMENT_STARTED=0
}

prepare_failed_deployment() {
  if [ "$SERVICES_STOPPED" -eq 1 ]; then
    systemctl stop sightindex-attribute-backfill.service 2>/dev/null || true
    systemctl stop sightindex-embedding.service 2>/dev/null || true
    systemctl stop sightindex-api.service 2>/dev/null || true
    systemctl stop sightindex-reid.service 2>/dev/null || true
  fi
  restore_previous_venv
  restore_previous_frontend
}

fail() {
  trap - ERR
  printf '[sightindex-deploy] ERROR: %s\n' "$*" >&2
  prepare_failed_deployment
  failure_hint
  exit 1
}

on_error() {
  local exit_code="$1"
  local line="$2"
  trap - ERR
  printf '[sightindex-deploy] ERROR: command failed at line %s (exit %s)\n' \
    "$line" "$exit_code" >&2
  prepare_failed_deployment
  failure_hint
  exit "$exit_code"
}

trap 'on_error "$?" "$LINENO"' ERR

while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-deps) SKIP_DEPS=1 ;;
    --skip-frontend) SKIP_FRONTEND=1 ;;
    --skip-milvus) SKIP_MILVUS=1 ;;
    --skip-backup) SKIP_BACKUP=1 ;;
    --start-backfill) START_BACKFILL=1 ;;
    --skip-verify) SKIP_VERIFY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; fail "unknown option: $1" ;;
  esac
  shift
done

[ "$(id -u)" -eq 0 ] || fail "run as root so systemd units can be installed"
[ "$ROOT_DIR" = "$EXPECTED_ROOT" ] \
  || fail "systemd units expect $EXPECTED_ROOT; current checkout is $ROOT_DIR"
cd "$ROOT_DIR"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"
python3 - <<'PY' || fail "Python 3.11 or newer is required"
import sys

raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
command -v runuser >/dev/null 2>&1 || fail "runuser is required"
command -v systemctl >/dev/null 2>&1 || fail "systemd is required"
command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v flock >/dev/null 2>&1 || fail "flock is required"
id "$SERVICE_USER" >/dev/null 2>&1 \
  || fail "service account '$SERVICE_USER' is missing; follow docs/deployment.md first"
id "$DEPLOY_USER" >/dev/null 2>&1 \
  || fail "deployment account '$DEPLOY_USER' is missing; follow docs/deployment.md first"
getent group "$SERVICE_GROUP" >/dev/null 2>&1 \
  || fail "service group '$SERVICE_GROUP' is missing"
[ "$(id -gn "$DEPLOY_USER")" = "$SERVICE_GROUP" ] \
  || fail "$DEPLOY_USER must use $SERVICE_GROUP as its primary group"
[ "$(stat -c '%U' "$ROOT_DIR")" = "$DEPLOY_USER" ] \
  || fail "$ROOT_DIR must be owned by the trusted deployment account $DEPLOY_USER"
if runuser -u "$SERVICE_USER" -- env -i \
  "HOME=$SERVICE_HOME" \
  "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  test -w "$ROOT_DIR"; then
  fail "$SERVICE_USER must not be able to modify the deployment checkout"
fi

[ ! -L "$SERVICE_HOME" ] || fail "$SERVICE_HOME must not be a symbolic link"
install -d -m 0750 -o root -g "$SERVICE_GROUP" "$SERVICE_HOME"
[ "$(readlink -f "$SERVICE_HOME")" = "$SERVICE_HOME" ] \
  || fail "$SERVICE_HOME must resolve to itself"
[ ! -L "$BACKUP_DIR" ] || fail "$BACKUP_DIR must not be a symbolic link"
[ ! -L "$SERVICE_HOME/.cache" ] || fail "$SERVICE_HOME/.cache must not be a symbolic link"

install -d -m 0700 -o root -g root "$LOG_DIR" "$BACKUP_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || fail "another deployment is already running"

if [ ! -f "$ENV_FILE" ]; then
  [ ! -e "$ENV_FILE" ] && [ ! -L "$ENV_FILE" ] \
    || fail "$ENV_FILE exists but is not a regular file"
  install -o root -g "$SERVICE_GROUP" -m 0640 "$ENV_TEMPLATE" "$ENV_FILE"
  fail "created $ENV_FILE from the RTX 5090 template; review every value and rerun"
fi
[ ! -L "$ENV_FILE" ] || fail "$ENV_FILE must not be a symbolic link"
[ -f "$ENV_FILE" ] || fail "$ENV_FILE must be a regular file"
[ "$(stat -c '%U' "$ENV_FILE")" = root ] \
  || fail "$ENV_FILE must already be owned by root; secure it manually before deployment"
[ "$(stat -c '%h' "$ENV_FILE")" -eq 1 ] \
  || fail "$ENV_FILE must not have additional hard links"
if runuser -u "$SERVICE_USER" -- test -w "$ENV_FILE"; then
  fail "$ENV_FILE must not be writable by $SERVICE_USER; secure it manually before deployment"
fi
chown "root:$SERVICE_GROUP" "$ENV_FILE"
chmod 0640 "$ENV_FILE"

# Parse the environment file as data. Never source a service-writable file from this root
# process: shell evaluation would turn a deployment setting into a privilege-escalation path.
parsed_env="$(mktemp)"
if ! python3 - "$ENV_FILE" >"$parsed_env" <<'PY'
import re
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1])
seen: set[str] = set()
for line_number, original in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    stripped = original.strip()
    if not stripped or stripped.startswith("#"):
        continue
    if stripped.startswith("export "):
        raise SystemExit(f"{path}:{line_number}: use KEY=value without export")
    key, separator, raw_value = stripped.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise SystemExit(f"{path}:{line_number}: expected KEY=value without whitespace")
    if raw_value[:1].isspace():
        raise SystemExit(f"{path}:{line_number}: value must begin immediately after =")
    if key in {
        "BASHOPTS", "BASH_ENV", "CDPATH", "ENV", "GLOBIGNORE", "HOME", "IFS",
        "LOGNAME", "PATH", "SHELL", "SHELLOPTS", "USER",
    }:
        raise SystemExit(f"{path}:{line_number}: reserved process variable {key}")
    if key in seen:
        raise SystemExit(f"{path}:{line_number}: duplicate key {key}")
    if any(character in raw_value for character in "$`\\;|&<>()#~"):
        raise SystemExit(
            f"{path}:{line_number}: shell expansion and control characters are not supported"
        )
    lexer = shlex.shlex(raw_value, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    parts = list(lexer)
    if len(parts) > 1:
        raise SystemExit(f"{path}:{line_number}: quote values containing spaces")
    value = parts[0] if parts else ""
    if any(character in value for character in "\t\r\n"):
        raise SystemExit(f"{path}:{line_number}: tabs and newlines are not supported")
    seen.add(key)
    print(f"{key}\t{value}")
PY
then
  rm -f "$parsed_env"
  fail "$ENV_FILE is not a valid deployment environment file"
fi
while IFS=$'\t' read -r key value; do
  printf -v "CONFIG_$key" '%s' "$value"
done <"$parsed_env"
rm -f "$parsed_env"

cfg() {
  local key="$1"
  local default_value="${2-}"
  local variable_name="CONFIG_$key"
  if [ "${!variable_name+x}" = x ]; then
    printf '%s' "${!variable_name}"
  else
    printf '%s' "$default_value"
  fi
}

has_cfg() {
  local variable_name="CONFIG_$1"
  [ "${!variable_name+x}" = x ]
}

is_true() {
  case "$1" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

absolute_path() {
  local candidate="$1"
  if [[ "$candidate" = /* ]]; then
    printf '%s\n' "$candidate"
  else
    printf '%s/%s\n' "$ROOT_DIR" "${candidate#./}"
  fi
}

canonical_path() {
  python3 -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve(strict=False))' \
    "$1"
}

as_service() {
  runuser -u "$SERVICE_USER" -- env -i \
    "HOME=$SERVICE_HOME" \
    "USER=$SERVICE_USER" \
    "LOGNAME=$SERVICE_USER" \
    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    "LANG=C" \
    "$@"
}

as_service_configured() {
  runuser -u "$SERVICE_USER" -- env -i \
    "HOME=$SERVICE_HOME" \
    "USER=$SERVICE_USER" \
    "LOGNAME=$SERVICE_USER" \
    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    "LANG=C" \
    bash -c '
      env_file="$1"
      shift
      set -a
      . "$env_file"
      set +a
      exec "$@"
    ' bash "$ENV_FILE" "$@"
}

as_deployer() {
  runuser -u "$DEPLOY_USER" -- env -i \
    "HOME=$DEPLOY_HOME" \
    "USER=$DEPLOY_USER" \
    "LOGNAME=$DEPLOY_USER" \
    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    "LANG=C" \
    bash -c 'umask 0027; exec "$@"' bash "$@"
}

assert_trusted_artifact() {
  local artifact_path="$1"
  local artifact_name="$2"
  local untrusted_owner writable_path
  [ -e "$artifact_path" ] || fail "$artifact_name is missing: $artifact_path"
  [ ! -L "$artifact_path" ] || fail "$artifact_name must not be a symbolic link"
  untrusted_owner="$(find "$artifact_path" -xdev ! -user "$DEPLOY_USER" -print -quit)"
  [ -z "$untrusted_owner" ] \
    || fail "$artifact_name contains a path not owned by $DEPLOY_USER: $untrusted_owner"
  writable_path="$(as_service find "$artifact_path" -xdev -writable -print -quit)"
  [ -z "$writable_path" ] \
    || fail "$artifact_name contains a path writable by $SERVICE_USER: $writable_path"
}

app_host="$(cfg APP_HOST 127.0.0.1)"
app_port="$(cfg APP_PORT 8000)"
public_base_url="$(cfg PUBLIC_BASE_URL)"
database_url="$(cfg DATABASE_URL sqlite:////var/lib/sightindex/db/sightindex.db)"
data_dir="$(canonical_path "$(absolute_path "$(cfg DATA_DIR /var/lib/sightindex/data)")")"
milvus_bind="$(cfg MILVUS_BIND 127.0.0.1)"
milvus_metric="$(cfg MILVUS_METRIC_TYPE COSINE)"
minio_user="$(cfg MINIO_ROOT_USER)"
minio_password="$(cfg MINIO_ROOT_PASSWORD)"
face_root="$(absolute_path "$(cfg FACE_INSIGHTFACE_ROOT /var/lib/sightindex/models/insightface)")"
face_model="$(cfg FACE_INSIGHTFACE_MODEL buffalo_l)"
reid_checkpoint_dir="$(absolute_path "$(cfg REID_CHECKPOINT_DIR /var/lib/sightindex/models/sapiensid_wb12m)")"
reid_pose_weights="$(absolute_path "$(cfg REID_POSE_WEIGHTS /var/lib/sightindex/.cache/yolov8n-pose.pt)")"

has_cfg APP_HOST && [ -n "$(cfg APP_HOST)" ] \
  || fail "APP_HOST must be explicitly set in $ENV_FILE"
has_cfg APP_PORT && [ -n "$(cfg APP_PORT)" ] \
  || fail "APP_PORT must be explicitly set in $ENV_FILE"
is_true "$(cfg REID_ENABLED false)" || fail "REID_ENABLED must be true for this profile"
is_true "$(cfg MILVUS_ENABLED false)" || fail "MILVUS_ENABLED must be true for this profile"
[ "$milvus_metric" = "COSINE" ] || fail "MILVUS_METRIC_TYPE must be COSINE for ReID"
[ "$app_host" = "127.0.0.1" ] \
  || fail "APP_HOST must be 127.0.0.1; expose the service through a TLS reverse proxy"
case "$public_base_url" in
  https://*.example.com|https://*.example.test|"")
    fail "replace PUBLIC_BASE_URL with the deployment's real HTTPS URL"
    ;;
  https://*) ;;
  *) fail "PUBLIC_BASE_URL must use HTTPS" ;;
esac
[[ "$app_port" =~ ^[0-9]+$ ]] && [ "$app_port" -ge 1 ] && [ "$app_port" -le 65535 ] \
  || fail "APP_PORT must be an integer between 1 and 65535"
[ "$milvus_bind" = "127.0.0.1" ] \
  || fail "MILVUS_BIND must be 127.0.0.1 for the single-host profile"
[ "$(cfg MILVUS_HOST 127.0.0.1)" = "127.0.0.1" ] \
  || fail "MILVUS_HOST must be 127.0.0.1 for the single-host profile"
[ "$(cfg REID_SERVICE_HOST 127.0.0.1)" = "127.0.0.1" ] \
  || fail "REID_SERVICE_HOST must be 127.0.0.1 for the single-host profile"
[ "$(cfg REID_SERVICE_URL http://127.0.0.1:18031)" = \
  "http://127.0.0.1:$(cfg REID_SERVICE_PORT 18031)" ] \
  || fail "REID_SERVICE_URL must match the local REID_SERVICE_PORT"
[ -n "$minio_user" ] || fail "MINIO_ROOT_USER must be set"
[ "${#minio_password}" -ge 16 ] \
  || fail "MINIO_ROOT_PASSWORD must contain at least 16 characters"
case "$minio_password" in
  *replace-with-random-secret*|*change-me-before-use*|minioadmin)
    fail "replace the placeholder MINIO_ROOT_PASSWORD"
    ;;
esac
[ "$(cfg FACE_EMBEDDING_PROVIDER insightface)" = "insightface" ] \
  || fail "FACE_EMBEDDING_PROVIDER must be insightface"
[ "$face_model" = "buffalo_l" ] || fail "FACE_INSIGHTFACE_MODEL must be buffalo_l"
is_true "$(cfg FACE_INSIGHTFACE_ALLOW_DOWNLOAD false)" \
  && fail "FACE_INSIGHTFACE_ALLOW_DOWNLOAD must remain false in production"
if [ -n "$(cfg APP_BASIC_AUTH_USERNAME)" ] || [ -n "$(cfg APP_BASIC_AUTH_PASSWORD)" ]; then
  [ -n "$(cfg APP_BASIC_AUTH_USERNAME)" ] && [ -n "$(cfg APP_BASIC_AUTH_PASSWORD)" ] \
    || fail "set both APP_BASIC_AUTH_USERNAME and APP_BASIC_AUTH_PASSWORD, or neither"
  [[ "$(cfg APP_BASIC_AUTH_USERNAME)" != *:* ]] \
    || fail "APP_BASIC_AUTH_USERNAME cannot contain a colon"
fi
if [ "$START_BACKFILL" -eq 1 ]; then
  [ "$(cfg VLM_PROVIDER none)" != "none" ] \
    || fail "configure VLM_PROVIDER before using --start-backfill"
  [ "$(cfg VLM_MODEL your-vlm-model)" != "your-vlm-model" ] \
    || fail "replace the VLM_MODEL placeholder before using --start-backfill"
fi

[ "$(uname -m)" = "x86_64" ] || fail "the RTX 5090 profile requires an x86_64 host"
[[ "$database_url" != *:memory:* ]] || fail "an in-memory database is not a deployment target"
[ "$data_dir" = "$SERVICE_HOME/data" ] \
  || fail "DATA_DIR must resolve exactly to $SERVICE_HOME/data for this profile"
if [[ "$database_url" = postgresql* ]]; then
  [ "$(cfg POSTGRES_BIND 127.0.0.1)" = "127.0.0.1" ] \
    || fail "POSTGRES_BIND must be 127.0.0.1"
fi
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q '5090' \
  || fail "nvidia-smi did not report an RTX 5090"

as_service test -r "$ROOT_DIR/requirements.rtx5090.txt" \
  || fail "$SERVICE_USER cannot read the checkout"
if as_service test -w "$ROOT_DIR"; then
  fail "$SERVICE_USER must not be able to modify the deployment checkout"
fi
writable_checkout_path="$(as_service find "$ROOT_DIR" -xdev \
  \( -path "$ROOT_DIR/.venv" -o -path "$ROOT_DIR/.venv/*" \
     -o -path "$ROOT_DIR/.venv.previous" -o -path "$ROOT_DIR/.venv.previous/*" \
     -o -path "$ROOT_DIR/frontend/node_modules" -o -path "$ROOT_DIR/frontend/node_modules/*" \
     -o -path "$ROOT_DIR/frontend/dist" -o -path "$ROOT_DIR/frontend/dist/*" \
     -o -path "$ROOT_DIR/frontend/dist.previous" -o -path "$ROOT_DIR/frontend/dist.previous/*" \) \
  -prune -o -writable -print -quit)"
[ -z "$writable_checkout_path" ] \
  || fail "$SERVICE_USER can modify a source/configuration path inside the deployment checkout: $writable_checkout_path"
as_deployer test -w "$ROOT_DIR" \
  || fail "$DEPLOY_USER cannot update the deployment checkout"
working_tree_status="$(as_deployer git -C "$ROOT_DIR" status --porcelain --untracked-files=all)"
if [ -n "$working_tree_status" ]; then
  fail "deployment checkout is not clean; review tracked and untracked files before deployment"
fi
install -d -m 0700 -o "$SERVICE_USER" -g "$SERVICE_GROUP" \
  "$data_dir" "$SERVICE_HOME/.cache"
as_service test -w "$data_dir" || fail "$SERVICE_USER cannot write DATA_DIR=$data_dir"
if [[ "$database_url" = sqlite:///* ]]; then
  sqlite_database_path="$(canonical_path "$(absolute_path "${database_url#sqlite:///}")")"
  [ "$(dirname "$sqlite_database_path")" = "$SERVICE_HOME/db" ] \
    || fail "the RTX SQLite database must be a direct child of $SERVICE_HOME/db"
  install -d -m 0700 -o "$SERVICE_USER" -g "$SERVICE_GROUP" \
    "$SERVICE_HOME/db"
fi

face_model_dir="$face_root/models/$face_model"
for face_asset in det_10g.onnx w600k_r50.onnx; do
  asset="$face_model_dir/$face_asset"
  [ -f "$asset" ] || fail "InsightFace asset missing: $asset"
  as_service test -r "$asset" || fail "$SERVICE_USER cannot read $asset"
done

for reid_asset in model.pth model.yaml; do
  asset="$reid_checkpoint_dir/$reid_asset"
  [ -f "$asset" ] || fail "SapiensID asset missing: $asset"
  as_service test -r "$asset" || fail "$SERVICE_USER cannot read $asset"
done
reid_vendor_src="$ROOT_DIR/deploy/agx/reid_service/sapiensid/tasks/sapiensID/src"
for asset in \
  "$reid_vendor_src/aligners/configs/yolo_dfa.yaml" \
  "$reid_vendor_src/aligners/keypoint_predictor/pretrained_models/aligners/dfa_mobilenetv4_medium/mobilenetv4_Final.pth" \
  "$reid_pose_weights"; do
  [ -f "$asset" ] || fail "SapiensID alignment asset missing: $asset"
  as_service test -r "$asset" || fail "$SERVICE_USER cannot read $asset"
done
[ "$reid_pose_weights" = "$SERVICE_HOME/.cache/yolov8n-pose.pt" ] \
  || fail "the ReID server reads $SERVICE_HOME/.cache/yolov8n-pose.pt; set REID_POSE_WEIGHTS to that path"

if [ "$(cfg PERSON_DETECTOR yolo)" = "yolo" ]; then
  yolo_model="$(absolute_path "$(cfg YOLO_MODEL)")"
  [ -n "$(cfg YOLO_MODEL)" ] || fail "YOLO_MODEL must be set"
  [ -f "$yolo_model" ] || fail "YOLO model missing: $yolo_model"
  as_service test -r "$yolo_model" || fail "$SERVICE_USER cannot read $yolo_model"
fi

if [ "$SKIP_FRONTEND" -eq 0 ]; then
  command -v npm >/dev/null 2>&1 || fail "npm is required to build the console"
  command -v node >/dev/null 2>&1 || fail "Node.js is required to build the console"
  node -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 22 || (major === 22 && minor >= 18) ? 0 : 1)' \
    || fail "Node.js 22.18 or newer is required"
fi
if [ "$SKIP_DEPS" -eq 1 ]; then
  assert_trusted_artifact "$ROOT_DIR/.venv" "existing Python environment"
  as_service test -x "$ROOT_DIR/.venv/bin/python" \
    || fail "$SERVICE_USER cannot execute the existing Python environment"
fi
if [ "$SKIP_FRONTEND" -eq 1 ]; then
  assert_trusted_artifact "$ROOT_DIR/frontend/dist" "existing frontend bundle"
  as_service test -r "$ROOT_DIR/frontend/dist/index.html" \
    || fail "$SERVICE_USER cannot read the existing frontend bundle"
fi
if [ "$SKIP_MILVUS" -eq 0 ]; then
  command -v docker >/dev/null 2>&1 || fail "Docker with Compose v2 is required for Milvus"
  docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required"
  docker compose --env-file "$ENV_FILE" -f deploy/milvus/docker-compose.yml config --quiet \
    || fail "Milvus Compose configuration is invalid"
fi

backup_database() {
  local timestamp backup_path temporary_path database_path
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  if [[ "$database_url" = sqlite:///* ]]; then
    database_path="${database_url#sqlite:///}"
    database_path="$(absolute_path "$database_path")"
    if [ ! -f "$database_path" ]; then
      log "SQLite database is not present yet; no database backup is needed"
      return
    fi
    backup_path="$BACKUP_DIR/sightindex-$timestamp.db"
    temporary_path="$backup_path.partial"
    install -m 0600 /dev/null "$temporary_path"
    log "creating a SQLite online backup at $backup_path"
    python3 - "$database_path" "$temporary_path" <<'PY'
import sqlite3
import sys

source = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
target = sqlite3.connect(sys.argv[2])
with target:
    source.backup(target)
target.close()
source.close()
PY
    mv "$temporary_path" "$backup_path"
    return
  fi

  if [[ "$database_url" = postgresql* ]]; then
    command -v docker >/dev/null 2>&1 \
      || fail "Docker is required to back up the repository-managed PostgreSQL service"
    if ! docker compose --env-file "$ENV_FILE" ps --status running --services \
      | grep -Fxq postgres; then
      fail "PostgreSQL is not running in the repository Compose stack; create an external backup and rerun with --skip-backup"
    fi
    if ! printf '%s\n' "$database_url" | python3 -c '
import sys
from urllib.parse import unquote, urlsplit

expected_user, expected_database, expected_port = sys.argv[1:]
parsed = urlsplit(sys.stdin.readline().strip())
valid = (
    parsed.scheme.startswith("postgresql")
    and parsed.hostname == "127.0.0.1"
    and (parsed.port or 5432) == int(expected_port)
    and unquote(parsed.username or "") == expected_user
    and unquote(parsed.path.lstrip("/")) == expected_database
)
raise SystemExit(0 if valid else 1)
' "$(cfg POSTGRES_USER sightindex)" "$(cfg POSTGRES_DB sightindex)" \
      "$(cfg POSTGRES_PORT 5432)"; then
      fail "DATABASE_URL does not target the running repository PostgreSQL service; create an external backup and rerun with --skip-backup"
    fi
    backup_path="$BACKUP_DIR/sightindex-$timestamp.dump"
    temporary_path="$backup_path.partial"
    install -m 0600 /dev/null "$temporary_path"
    log "creating a PostgreSQL backup at $backup_path"
    docker compose --env-file "$ENV_FILE" exec -T postgres \
      pg_dump -U "$(cfg POSTGRES_USER sightindex)" -d "$(cfg POSTGRES_DB sightindex)" -Fc \
      >"$temporary_path"
    test -s "$temporary_path" || fail "PostgreSQL backup is empty"
    mv "$temporary_path" "$backup_path"
    return
  fi

  fail "automatic backup does not support the configured database driver; back it up externally and use --skip-backup"
}

# Stop writers and GPU services before mutating the shared virtual environment. On failure they
# deliberately stay stopped so a partially updated environment is never restarted automatically.
log "stopping application services for the update"
if systemctl is-active --quiet sightindex-attribute-backfill.service; then
  BACKFILL_WAS_ACTIVE=1
fi
systemctl is-active --quiet sightindex-embedding.service \
  && fail "the optional embedding service uses separate dependencies; stop and disable it before applying the RTX ReID profile"
systemctl is-enabled --quiet sightindex-embedding.service \
  && fail "disable sightindex-embedding.service before applying the RTX ReID profile"
SERVICES_STOPPED=1
stop_service() {
  local service_name="$1"
  systemctl stop "$service_name" >/dev/null 2>&1 || true
  if systemctl is-active --quiet "$service_name"; then
    fail "$service_name is still active; refusing to mutate the deployment"
  fi
}
stop_service sightindex-attribute-backfill.service
stop_service sightindex-embedding.service
stop_service sightindex-api.service
stop_service sightindex-reid.service

if [ "$SKIP_BACKUP" -eq 0 ]; then
  backup_database
else
  log "database backup skipped by operator request"
fi

if [ "$SKIP_DEPS" -eq 0 ]; then
  log "building a new Python virtual environment"
  rm -rf "$ROOT_DIR/.venv.rollback"
  if [ -d "$ROOT_DIR/.venv" ]; then
    mv "$ROOT_DIR/.venv" "$ROOT_DIR/.venv.rollback"
  fi
  VENV_REPLACEMENT_STARTED=1
  as_deployer python3 -m venv "$ROOT_DIR/.venv"
  python_bin="$ROOT_DIR/.venv/bin/python"
  log "installing RTX 5090 Python dependencies (logs: $LOG_DIR)"
  as_deployer "$python_bin" -m pip install --upgrade pip setuptools wheel \
    >"$LOG_DIR/pip-bootstrap.log" 2>&1
  pytorch_index_url="$(cfg PYTORCH_INDEX_URL https://download.pytorch.org/whl/cu128)"
  case "$pytorch_index_url" in
    https://download.pytorch.org/whl/*) ;;
    *) fail "PYTORCH_INDEX_URL must use the official download.pytorch.org wheel index" ;;
  esac
  as_deployer "$python_bin" -m pip install torch torchvision --index-url "$pytorch_index_url" \
    >"$LOG_DIR/pip-torch.log" 2>&1
  as_deployer "$python_bin" -m pip install -r "$ROOT_DIR/requirements.rtx5090.txt" \
    >"$LOG_DIR/pip-install.log" 2>&1
  as_deployer "$python_bin" -m pip check >>"$LOG_DIR/pip-install.log" 2>&1
else
  [ -x "$ROOT_DIR/.venv/bin/python" ] \
    || fail "--skip-deps requires an existing $ROOT_DIR/.venv"
  python_bin="$ROOT_DIR/.venv/bin/python"
fi
assert_trusted_artifact "$ROOT_DIR/.venv" "Python environment"
as_service test -x "$python_bin" || fail "$SERVICE_USER cannot execute $python_bin"

log "checking CUDA runtimes"
as_service_configured "$python_bin" - <<'PY'
import onnxruntime as ort
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA")
device = torch.cuda.get_device_name(0)
if "5090" not in device:
    raise SystemExit(f"expected RTX 5090, got {device!r}")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("onnxruntime-gpu does not expose CUDAExecutionProvider")
value = (torch.ones(1024, device="cuda") * 2).sum().item()
torch.cuda.synchronize()
if value != 2048:
    raise SystemExit("unexpected CUDA tensor result")
print(
    f"PyTorch {torch.__version__} / CUDA {torch.version.cuda}: {device}; "
    "ONNX Runtime: CUDAExecutionProvider"
)
PY
as_service "$python_bin" -m pip freeze >"$LOG_DIR/python-environment.txt"
nvidia-smi >"$LOG_DIR/nvidia-smi.txt"

log "validating application configuration"
as_service_configured "$python_bin" - <<'PY'
from pydantic import ValidationError

from app.config.settings import Settings

try:
    Settings()
except ValidationError as exc:
    fields = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
    raise SystemExit(f"invalid SightIndex settings: {', '.join(fields)}") from None
print("SightIndex settings: valid")
PY

if [ "$SKIP_FRONTEND" -eq 0 ]; then
  log "building frontend"
  rm -rf "$ROOT_DIR/frontend/dist.rollback"
  if [ -d "$ROOT_DIR/frontend/dist" ]; then
    mv "$ROOT_DIR/frontend/dist" "$ROOT_DIR/frontend/dist.rollback"
  fi
  FRONTEND_REPLACEMENT_STARTED=1
  as_deployer npm --prefix "$ROOT_DIR/frontend" ci >"$LOG_DIR/frontend-build.log" 2>&1
  as_deployer npm --prefix "$ROOT_DIR/frontend" run build >>"$LOG_DIR/frontend-build.log" 2>&1
fi
[ -f "$ROOT_DIR/frontend/dist/index.html" ] || fail "frontend/dist/index.html is missing"
assert_trusted_artifact "$ROOT_DIR/frontend/dist" "frontend bundle"
as_service test -r "$ROOT_DIR/frontend/dist/index.html" \
  || fail "$SERVICE_USER cannot read the frontend bundle"

if [ "$SKIP_MILVUS" -eq 0 ]; then
  log "starting Milvus"
  docker compose --env-file "$ENV_FILE" -f deploy/milvus/docker-compose.yml up -d
  milvus_ready=0
  for ((attempt = 1; attempt <= 60; attempt += 1)); do
    if curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:9091/healthz \
      >/dev/null 2>&1; then
      milvus_ready=1
      break
    fi
    sleep 2
  done
  [ "$milvus_ready" -eq 1 ] || fail "Milvus did not become healthy within 120 seconds"
fi

log "installing explicit systemd units"
for unit_name in \
  sightindex-api.service \
  sightindex-reid.service \
  sightindex-embedding.service \
  sightindex-attribute-backfill.service; do
  install -m 0644 "$ROOT_DIR/deploy/systemd/$unit_name" "/etc/systemd/system/$unit_name"
done
systemctl daemon-reload
systemctl enable sightindex-reid.service sightindex-api.service >/dev/null

log "starting ReID model service"
systemctl restart sightindex-reid.service
reid_ready=0
for ((attempt = 1; attempt <= 150; attempt += 1)); do
  if curl --noproxy '*' -fsS --max-time 2 \
    "http://127.0.0.1:$(cfg REID_SERVICE_PORT 18031)/ready" \
    >/dev/null 2>&1; then
    reid_ready=1
    break
  fi
  sleep 2
done
[ "$reid_ready" -eq 1 ] \
  || fail "ReID did not become ready within 300 seconds; inspect journalctl -u sightindex-reid"

log "starting API and console"
systemctl restart sightindex-api.service
api_ready=0
for ((attempt = 1; attempt <= 60; attempt += 1)); do
  if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$app_port/health" \
    >/dev/null 2>&1; then
    api_ready=1
    break
  fi
  sleep 2
done
[ "$api_ready" -eq 1 ] \
  || fail "API did not become healthy within 120 seconds; inspect journalctl -u sightindex-api"

if [ "$START_BACKFILL" -eq 1 ] || [ "$BACKFILL_WAS_ACTIVE" -eq 1 ]; then
  log "starting or resuming structured-attribute backfill"
  systemctl enable sightindex-attribute-backfill.service >/dev/null
  systemctl restart sightindex-attribute-backfill.service
fi

if [ "$SKIP_VERIFY" -eq 0 ]; then
  as_service_configured bash "$ROOT_DIR/deploy/rtx5090/verify.sh"
fi

if [ "$VENV_REPLACEMENT_STARTED" -eq 1 ]; then
  rm -rf "$ROOT_DIR/.venv.previous"
  if [ -d "$ROOT_DIR/.venv.rollback" ]; then
    mv "$ROOT_DIR/.venv.rollback" "$ROOT_DIR/.venv.previous"
  fi
  VENV_REPLACEMENT_STARTED=0
fi
if [ "$FRONTEND_REPLACEMENT_STARTED" -eq 1 ]; then
  rm -rf "$ROOT_DIR/frontend/dist.previous"
  if [ -d "$ROOT_DIR/frontend/dist.rollback" ]; then
    mv "$ROOT_DIR/frontend/dist.rollback" "$ROOT_DIR/frontend/dist.previous"
  fi
  FRONTEND_REPLACEMENT_STARTED=0
fi

SERVICES_STOPPED=0
log "deployment completed at revision $(as_deployer git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || printf unknown)"
