#!/usr/bin/env bash
# One-command installer for the SightIndex container stack.
#
# Wraps deploy.sh so a fresh machine reaches a healthy API without the manual
# checklist: preflight checks (docker/compose, RAM, disk, GPU), mirror defaults
# for restricted networks, offline installs from a prebuilt bundle, then the
# normal deploy.sh path (release registration, image build, manage.sh up).
#
# Usage:
#   install.sh [--root DIR] [--stacks "base [reid]"] [--source DIR]
#              [--pip-mirror URL] [--npm-mirror URL] [--no-mirror]
#              [--offline BUNDLE.tar]
#
#   --root        deployment root (default $SIGHTINDEX_ROOT or /data/sightindex-bj-test)
#   --stacks      stacks to bring up (default: base; add reid on GPU machines)
#   --source      source checkout to deploy (default: the checkout holding this script)
#   --pip-mirror  PyPI mirror forwarded to the image build (default: Tsinghua mirror)
#   --npm-mirror  npm registry forwarded to the image build (default: npmmirror.com)
#   --no-mirror   build against upstream pypi.org / registry.npmjs.org
#   --offline     install from a bundle produced by make_offline_bundle.sh; no
#                 network beyond the bundle, images are docker-loaded, models
#                 copied, build skipped
#
# Examples:
#   bash deploy/containers/install.sh                          # online, CN mirrors
#   bash deploy/containers/install.sh --stacks "base reid"     # GPU machine
#   bash deploy/containers/install.sh --offline sightindex-offline-20260927.tar

set -euo pipefail

ROOT="${SIGHTINDEX_ROOT:-/data/sightindex-bj-test}"
STACKS="base"
SOURCE=""
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
NPM_MIRROR="https://registry.npmmirror.com"
OFFLINE_BUNDLE=""

usage() {
  sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --root)       [ $# -ge 2 ] || usage; ROOT="$2"; shift 2 ;;
    --stacks)     [ $# -ge 2 ] || usage; STACKS="$2"; shift 2 ;;
    --source)     [ $# -ge 2 ] || usage; SOURCE="$2"; shift 2 ;;
    --pip-mirror) [ $# -ge 2 ] || usage; PIP_MIRROR="$2"; shift 2 ;;
    --npm-mirror) [ $# -ge 2 ] || usage; NPM_MIRROR="$2"; shift 2 ;;
    --no-mirror)  PIP_MIRROR=""; NPM_MIRROR=""; shift ;;
    --offline)    [ $# -ge 2 ] || usage; OFFLINE_BUNDLE="$2"; shift 2 ;;
    -h|--help)    usage ;;
    *)            echo "Unknown argument: $1" >&2; usage ;;
  esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${SOURCE:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
[ -f "$SOURCE/deploy/containers/deploy.sh" ] || fail "no deploy.sh next to install.sh under $SOURCE"

ENV_FILE="$ROOT/.env"

env_value() {
  awk -F= -v key="$1" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$ENV_FILE" 2>/dev/null
}

set_env_value() {
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

# --- preflight --------------------------------------------------------------
echo "== preflight =="

command -v docker >/dev/null 2>&1 || fail "docker not found; install Docker Engine first"
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
else
  command -v docker-compose >/dev/null 2>&1 || fail "neither 'docker compose' nor 'docker-compose' available"
  COMPOSE="docker-compose"
fi
echo "docker + compose OK"

# The compose stack alone declares ~10 GiB of mem_limit for base, plus 6 GiB for reid.
required_mib=10240
case " $STACKS " in *" reid "*) required_mib=$((required_mib + 6144));; esac
if [ -r /proc/meminfo ]; then
  avail_mib="$(awk '/MemAvailable/ { printf "%d", $2 / 1024 }' /proc/meminfo)"
  if [ "$avail_mib" -lt $((required_mib * 60 / 100)) ]; then
    fail "only ${avail_mib} MiB RAM available; the '${STACKS}' stack needs ~${required_mib} MiB"
  elif [ "$avail_mib" -lt "$required_mib" ]; then
    warn "${avail_mib} MiB RAM available; ~${required_mib} MiB recommended for '${STACKS}'"
  else
    echo "RAM OK (${avail_mib} MiB available, ~${required_mib} MiB needed)"
  fi
fi

mkdir -p "$ROOT"
disk_min_mib=15360
disk_ok_mib=40960
if [ -n "$OFFLINE_BUNDLE" ]; then
  # The bundle is extracted under $ROOT before docker load consumes it.
  disk_min_mib=30720
  disk_ok_mib=61440
fi
disk_mib="$(df -Pm "$ROOT" | awk 'NR==2 { print $4 }')"
if [ "$disk_mib" -lt "$disk_min_mib" ]; then
  fail "only ${disk_mib} MiB free under $ROOT; leave at least $((disk_min_mib / 1024)) GiB (models, images, uploads)"
elif [ "$disk_mib" -lt "$disk_ok_mib" ]; then
  warn "${disk_mib} MiB free under $ROOT; $((disk_ok_mib / 1024)) GiB+ recommended when ingesting video"
else
  echo "disk OK (${disk_mib} MiB free under $ROOT)"
fi

case " $STACKS " in
  *" reid "*)
    command -v nvidia-smi >/dev/null 2>&1 || fail "reid stack requested but nvidia-smi not found"
    echo "GPU present: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
    ;;
esac

# --- offline: load bundle ---------------------------------------------------
DEPLOY_ARGS=(--root "$ROOT" --stacks "$STACKS")
if [ -n "$PIP_MIRROR" ]; then DEPLOY_ARGS+=(--pip-mirror "$PIP_MIRROR"); fi
if [ -n "$NPM_MIRROR" ]; then DEPLOY_ARGS+=(--npm-mirror "$NPM_MIRROR"); fi

if [ -n "$OFFLINE_BUNDLE" ]; then
  [ -f "$OFFLINE_BUNDLE" ] || fail "offline bundle not found: $OFFLINE_BUNDLE"
  echo
  echo "== offline install from $OFFLINE_BUNDLE =="
  STAGE="$(mktemp -d "$ROOT/offline-bundle.XXXXXX")"
  trap 'rm -rf "$STAGE"' EXIT
  tar -xf "$OFFLINE_BUNDLE" -C "$STAGE"
  [ -f "$STAGE/manifest.env" ] || fail "bundle is missing manifest.env; rebuild it with make_offline_bundle.sh"

  # shellcheck disable=SC1091
  . "$STAGE/manifest.env"
  [ -n "${SIGHTINDEX_IMAGE:-}" ] || fail "bundle manifest.env does not set SIGHTINDEX_IMAGE"
  [ -d "$STAGE/src" ] || fail "bundle is missing the src/ checkout"

  echo "Loading images (this can take a few minutes per image)"
  for image_tar in "$STAGE"/images/*.tar; do
    [ -e "$image_tar" ] || fail "bundle has no images/ directory"
    echo "  docker load < $(basename "$image_tar")"
    docker load -i "$image_tar"
  done
  docker image inspect "$SIGHTINDEX_IMAGE" >/dev/null 2>&1 \
    || fail "bundle image '$SIGHTINDEX_IMAGE' not present after docker load"

  if [ -d "$STAGE/models" ]; then
    echo "Copying models into $ROOT/models"
    mkdir -p "$ROOT/models"
    cp -a "$STAGE/models/." "$ROOT/models/"
  fi
  chown -R 1000:1000 "$ROOT/models" 2>/dev/null || true

  # deploy.sh never overwrites an existing env, so pre-create it with the
  # bundle's image reference; deploy.sh then only registers the release.
  if [ ! -f "$ENV_FILE" ]; then
    echo "Generating env file: $ENV_FILE"
    sed \
      -e "s#^MEDIA_DIR=.*#MEDIA_DIR=$ROOT/media#" \
      -e "s#^MODEL_DIR=.*#MODEL_DIR=$ROOT/models#" \
      -e "s#^CACHE_DIR=.*#CACHE_DIR=$ROOT/cache#" \
      "$STAGE/src/deploy/containers/.env.example" > "$ENV_FILE"
    for key in APP_BASIC_AUTH_PASSWORD POSTGRES_PASSWORD MINIO_ROOT_PASSWORD REID_SERVICE_API_KEY; do
      set_env_value "$key" "$(random_hex)"
    done
    set_env_value SIGHTINDEX_IMAGE "$SIGHTINDEX_IMAGE"
    if [ -n "${REID_CHECKPOINT_REVISION:-}" ]; then
      set_env_value REID_CHECKPOINT_REVISION "$REID_CHECKPOINT_REVISION"
    fi
    case " $STACKS " in *" reid "*) set_env_value REID_ENABLED true;; esac
    chmod 600 "$ENV_FILE"
  else
    echo "Env file exists, keeping it: $ENV_FILE"
    current_image="$(env_value SIGHTINDEX_IMAGE)"
    if [ -n "$current_image" ] && [ "$current_image" != "$SIGHTINDEX_IMAGE" ]; then
      # The only usable image offline is the bundle's; repoint (image refs are
      # not secrets, unlike everything else deploy.sh refuses to touch).
      set_env_value SIGHTINDEX_IMAGE "$SIGHTINDEX_IMAGE"
      echo "  SIGHTINDEX_IMAGE: $current_image -> $SIGHTINDEX_IMAGE (bundle image)"
    fi
  fi

  DEPLOY_ARGS+=(--source "$STAGE/src" --no-build)
else
  DEPLOY_ARGS+=(--source "$SOURCE")
fi

# --- deploy -----------------------------------------------------------------
echo
echo "== deploy (stacks: $STACKS) =="
bash "$SCRIPT_DIR/deploy.sh" "${DEPLOY_ARGS[@]}"

echo
echo "Install finished."
api_bind="$(env_value API_BIND)"; api_bind="${api_bind:-127.0.0.1}"
api_port="$(env_value API_PORT)"; api_port="${api_port:-18030}"
echo "  API:         http://$api_bind:$api_port"
echo "  credentials: $ENV_FILE -> APP_BASIC_AUTH_USERNAME / APP_BASIC_AUTH_PASSWORD"
echo "  operations:  bash $ROOT/manage.sh {status|logs|restart}"
