#!/usr/bin/env bash
# End-to-end deployment script for the SightIndex container stack.
#
# Turns a source checkout into a running deployment under one root directory:
# create the layout, generate the private env file (secrets, never overwrite
# an existing one), register the source as a release, build the application
# image, then drive `manage.sh up` and wait for API health.
#
# Usage:
#   deploy.sh [--root DIR] [--source DIR] [--release NAME] [--env-file FILE]
#             [--stacks "base [reid] [embedding] [semantic]"]
#             [--no-build] [--set-image]
#
#   --root       deployment root (default $SIGHTINDEX_ROOT or /data/sightindex-bj-test)
#   --source     SightIndex checkout containing deploy/containers/ (default: cwd)
#   --release    release name (default: <date>-<shortrev>); must not contain spaces
#   --env-file   env file (default: <root>/.env)
#   --stacks     stacks passed to manage.sh up (default: base)
#   --no-build   skip the image build; reuse the image named by SIGHTINDEX_IMAGE
#   --set-image  allow updating SIGHTINDEX_IMAGE in an existing env file
#
# Safety rules (mirrored from deploy/containers/README.md):
#   never overwrites an existing env file, never touches media/models/cache
#   or volumes, never runs `down -v`, and refuses the reid stack when the GPU
#   has less than 4 GiB free.

set -euo pipefail

ROOT="${SIGHTINDEX_ROOT:-/data/sightindex-bj-test}"
SOURCE="$(pwd)"
RELEASE=""
ENV_FILE=""
STACKS="base"
DO_BUILD=1
SET_IMAGE=0

usage() {
  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --root)      [ $# -ge 2 ] || usage; ROOT="$2"; shift 2 ;;
    --source)    [ $# -ge 2 ] || usage; SOURCE="$2"; shift 2 ;;
    --release)   [ $# -ge 2 ] || usage; RELEASE="$2"; shift 2 ;;
    --env-file)  [ $# -ge 2 ] || usage; ENV_FILE="$2"; shift 2 ;;
    --stacks)    [ $# -ge 2 ] || usage; STACKS="$2"; shift 2 ;;
    --no-build)  DO_BUILD=0; shift ;;
    --set-image) SET_IMAGE=1; shift ;;
    -h|--help)   usage ;;
    *)           echo "Unknown argument: $1" >&2; usage ;;
  esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }

# --- validate source and tooling -------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SOURCE/deploy/containers/compose.yaml" ] && \
   [ ! -f "$SOURCE/deploy/containers/compose.base.yaml" ]; then
  cat >&2 <<EOF
ERROR: --source (default: current directory) must be a SightIndex source
       checkout whose deploy/containers/ holds a base compose file; got: $SOURCE
EOF
  if [ -d "$SOURCE/releases" ] || [ -f "$SOURCE/manage.sh" ]; then
    cat >&2 <<EOF
       $SOURCE looks like an already-deployed root, not a source checkout.
       - day-to-day operations there:  bash $SOURCE/manage.sh {up|status|logs|restart}
       - re-deploying needs a source:  deploy.sh --source <SightIndex checkout> ...
EOF
  fi
  exit 1
fi

# Older releases were cut before manage.sh / .env.example existed in this
# directory; fall back to the copies shipped next to this script.
MANAGE_SRC="$SOURCE/deploy/containers/manage.sh"
[ -f "$MANAGE_SRC" ] || MANAGE_SRC="$SCRIPT_DIR/manage.sh"
[ -f "$MANAGE_SRC" ] || fail "manage.sh not found under $SOURCE/deploy/containers or next to deploy.sh ($SCRIPT_DIR)"

ENV_TEMPLATE="$SOURCE/deploy/containers/.env.example"
[ -f "$ENV_TEMPLATE" ] || ENV_TEMPLATE="$SCRIPT_DIR/.env.example"

if [ "$DO_BUILD" = 1 ] && [ ! -f "$SOURCE/deploy/containers/Dockerfile" ]; then
  fail "no deploy/containers/Dockerfile under $SOURCE (overlay-only release?); pass --no-build to reuse the image named by SIGHTINDEX_IMAGE"
fi
command -v docker >/dev/null 2>&1 || fail "docker not found"
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
else
  command -v docker-compose >/dev/null 2>&1 || fail "neither 'docker compose' nor 'docker-compose' available"
  COMPOSE="docker-compose"
fi

revision="unknown"
if git -C "$SOURCE" rev-parse --short HEAD >/dev/null 2>&1; then
  revision="$(git -C "$SOURCE" rev-parse --short HEAD)"
fi
RELEASE="${RELEASE:-$(date +%Y%m%d-%H%M%S)-${revision}}"
case "$RELEASE" in *\ *|*/*) fail "release name must not contain spaces or slashes: $RELEASE";; esac

ENV_FILE="${ENV_FILE:-$ROOT/.env}"

env_value() {
  awk -F= -v key="$1" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$ENV_FILE" 2>/dev/null
}

set_env_value() {
  # Update KEY in the env file, appending when missing; keeps everything else.
  local key="$1" value="$2" tmp
  tmp="$(mktemp)"
  awk -F= -v key="$key" -v val="$value" '
    BEGIN { done=0 }
    $1 == key && !done { print key "=" val; done=1; next }
    { print }
    END { if (!done) print key "=" val }
  ' "$ENV_FILE" > "$tmp"
  cat "$tmp" > "$ENV_FILE" && rm -f "$tmp"
}

random_hex() { head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n'; }

# --- layout -----------------------------------------------------------------
for dir in "$ROOT" "$ROOT/releases" "$ROOT/media" "$ROOT/models" "$ROOT/cache/api"; do
  if [ ! -d "$dir" ]; then
    mkdir -p "$dir"
    case "$dir" in
      "$ROOT/media"*) chown 1000:1000 "$dir" 2>/dev/null || \
        echo "NOTE: could not chown $dir to 1000:1000 (run as root, or fix manually)" >&2 ;;
      "$ROOT/cache"*)  chown 1000:1000 "$dir" 2>/dev/null || true ;;
    esac
  fi
done

# --- env file ---------------------------------------------------------------
if [ ! -f "$ENV_FILE" ]; then
  echo "Generating env file: $ENV_FILE"
  sed \
    -e "s#^MEDIA_DIR=.*#MEDIA_DIR=$ROOT/media#" \
    -e "s#^MODEL_DIR=.*#MODEL_DIR=$ROOT/models#" \
    -e "s#^CACHE_DIR=.*#CACHE_DIR=$ROOT/cache#" \
    "$ENV_TEMPLATE" > "$ENV_FILE"
  for key in APP_BASIC_AUTH_PASSWORD POSTGRES_PASSWORD MINIO_ROOT_PASSWORD REID_SERVICE_API_KEY; do
    set_env_value "$key" "$(random_hex)"
  done
  chmod 600 "$ENV_FILE"
  echo "  secrets generated (URL-safe hex). Review the file before exposing the API."
  GENERATED_ENV=1
else
  echo "Env file exists, keeping it: $ENV_FILE"
  GENERATED_ENV=0
fi

# --- register the release ---------------------------------------------------
RELEASE_DIR="$ROOT/releases/$RELEASE"
[ -e "$RELEASE_DIR" ] && fail "release already exists: $RELEASE_DIR"
mkdir -p "$RELEASE_DIR"
echo "Registering release: $RELEASE_DIR"
( cd "$SOURCE" && tar -cf - \
    --exclude=.git --exclude=node_modules --exclude=.venv --exclude=venv \
    --exclude=__pycache__ --exclude='*.pyc' --exclude=data \
    --exclude=test-results --exclude=.pytest_cache \
    . ) | ( cd "$RELEASE_DIR" && tar -xf - )

# Keep the operator entry point in sync with the deployed source.
install -m 0755 "$MANAGE_SRC" "$ROOT/manage.sh"

# --- image ------------------------------------------------------------------
IMAGE_TAG="sightindex:$RELEASE"
if [ "$DO_BUILD" = 1 ]; then
  echo "Building image: $IMAGE_TAG (context $SOURCE)"
  docker build -f "$SOURCE/deploy/containers/Dockerfile" \
    --build-arg SOURCE_REVISION="$revision" \
    -t "$IMAGE_TAG" "$SOURCE"
else
  echo "Skipping build (--no-build)."
fi

current_image="$(env_value SIGHTINDEX_IMAGE)"
if [ "$DO_BUILD" = 0 ]; then
  echo "--no-build: keeping SIGHTINDEX_IMAGE='$current_image'"
  if ! docker image inspect "$current_image" >/dev/null 2>&1; then
    echo "WARNING: image '$current_image' not present locally; 'up' will fail until it is built or pulled" >&2
  fi
elif [ -z "$current_image" ] || [ "$GENERATED_ENV" = 1 ] || [ "$SET_IMAGE" = 1 ]; then
  set_env_value SIGHTINDEX_IMAGE "$IMAGE_TAG"
  echo "SIGHTINDEX_IMAGE=$IMAGE_TAG"
else
  echo "SIGHTINDEX_IMAGE stays '$current_image' (pass --set-image to point it at $IMAGE_TAG)"
fi

# --- guards -----------------------------------------------------------------
case " $STACKS " in
  *" reid "*)
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      fail "reid stack requested but nvidia-smi not found"
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -cd '0-9')"
    [ -n "$free_mib" ] || fail "could not query GPU free memory"
    if [ "$free_mib" -lt 4096 ]; then
      fail "reid stack requested but GPU has only ${free_mib} MiB free (< 4096); keep REID_ENABLED=false"
    fi
    if [ "$(env_value REID_ENABLED)" != "true" ]; then
      fail "reid stack requested but REID_ENABLED is not 'true' in $ENV_FILE"
    fi
    ;;
esac
if [ ! -f "$(env_value MODEL_DIR)/yolo11n.pt" ]; then
  echo "WARNING: $(env_value MODEL_DIR)/yolo11n.pt missing -- API will run but person detection cannot;" \
       "prepare models per deploy/containers/README.md before ingesting media." >&2
fi

# --- start and verify -------------------------------------------------------
echo "Starting stacks via manage.sh: $STACKS"
bash "$ROOT/manage.sh" --env-file "$ENV_FILE" --release "$RELEASE_DIR" up $STACKS

bind="$(env_value API_BIND)"; bind="${bind:-127.0.0.1}"
port="$(env_value API_PORT)"; port="${port:-18030}"
health_url="http://$bind:$port/health"
echo "Waiting for API health: $health_url"
ok=0
for _ in $(seq 1 60); do
  if curl -fsS -m 5 "$health_url" >/dev/null 2>&1; then ok=1; break; fi
  sleep 3
done
if [ "$ok" != 1 ]; then
  echo "ERROR: API not healthy after 180s; inspect with: bash $ROOT/manage.sh --env-file $ENV_FILE logs api" >&2
  exit 1
fi
echo "API healthy: $health_url"
echo
echo "Deployed release: $RELEASE"
echo "Next: run the acceptance checks in deploy/containers/README.md"
echo "      (Basic Auth credentials: $ENV_FILE -> APP_BASIC_AUTH_USERNAME/PASSWORD)"
echo "      daily operations: bash $ROOT/manage.sh {status|logs|restart|down}"
