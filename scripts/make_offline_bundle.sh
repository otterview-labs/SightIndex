#!/usr/bin/env bash
# Build a self-contained offline installation bundle for SightIndex.
#
# Run this on a machine that already deployed via deploy.sh (images built,
# models dir populated). It snapshots the newest registered release as src/,
# docker-saves every image the compose file references, copies the models dir,
# and writes manifest.env + checksums. Ship the resulting .tar plus the emitted
# install.sh to the air-gapped machine and run:
#
#   bash install.sh --offline sightindex-offline-<date>.tar [--stacks "base reid"]
#
# Usage:
#   make_offline_bundle.sh [--root DIR] [--out DIR] [--release-dir DIR]
#
#   --root         deployment root to snapshot (default $SIGHTINDEX_ROOT or
#                  /data/sightindex-bj-test)
#   --out          directory receiving the bundle (default: <root>)
#   --release-dir  snapshot this release instead of the newest registered one
#
# The bundle is a plain tar (image layers are already compressed); expect it
# to need roughly image + model sizes, and the staging copy needs the same
# again while the script runs.

set -euo pipefail

ROOT="${SIGHTINDEX_ROOT:-/data/sightindex-bj-test}"
OUT=""
RELEASE_DIR=""

usage() {
  sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --root)        [ $# -ge 2 ] || usage; ROOT="$2"; shift 2 ;;
    --out)         [ $# -ge 2 ] || usage; OUT="$2"; shift 2 ;;
    --release-dir) [ $# -ge 2 ] || usage; RELEASE_DIR="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *)             echo "Unknown argument: $1" >&2; usage ;;
  esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }

ENV_FILE="$ROOT/.env"
[ -f "$ENV_FILE" ] || fail "no env file at $ENV_FILE; run deploy.sh on this machine first"
[ -d "$ROOT/releases" ] || fail "no releases/ under $ROOT; run deploy.sh on this machine first"

env_value() {
  awk -F= -v key="$1" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$ENV_FILE"
}

if [ -z "$RELEASE_DIR" ]; then
  for candidate in $(ls -1dt "$ROOT"/releases/*/ 2>/dev/null || true); do
    if [ -f "${candidate%/}/deploy/containers/compose.yaml" ] || \
       [ -f "${candidate%/}/deploy/containers/compose.base.yaml" ]; then
      RELEASE_DIR="${candidate%/}"
      break
    fi
  done
fi
[ -n "$RELEASE_DIR" ] && [ -d "$RELEASE_DIR" ] || fail "no release with a compose file under $ROOT/releases (or bad --release-dir)"
BASE_COMPOSE="$RELEASE_DIR/deploy/containers/compose.yaml"
[ -f "$BASE_COMPOSE" ] || BASE_COMPOSE="$RELEASE_DIR/deploy/containers/compose.base.yaml"
echo "Snapshotting release: $RELEASE_DIR"

MODEL_DIR="$(env_value MODEL_DIR)"
[ -n "$MODEL_DIR" ] && [ -d "$MODEL_DIR" ] || fail "MODEL_DIR from $ENV_FILE is missing or empty: ${MODEL_DIR:-<unset>}"
SIGHTINDEX_IMAGE="$(env_value SIGHTINDEX_IMAGE)"
[ -n "$SIGHTINDEX_IMAGE" ] || fail "SIGHTINDEX_IMAGE is not set in $ENV_FILE"

command -v docker >/dev/null 2>&1 || fail "docker not found"
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
else
  command -v docker-compose >/dev/null 2>&1 || fail "neither 'docker compose' nor 'docker-compose' available"
  COMPOSE="docker-compose"
fi

# Every image the deployment can reference, as compose resolves it (base file
# plus any overlays found in any release, all profiles enabled -- the superset
# is what an offline target may need to `up`).
COMPOSE_FILES=(-f "$BASE_COMPOSE")
for overlay in compose.embedding.yaml compose.semantic-search.yaml; do
  for candidate in "$RELEASE_DIR" $(ls -1dt "$ROOT"/releases/*/ 2>/dev/null | sed 's:/$::'); do
    if [ -f "$candidate/deploy/containers/$overlay" ]; then
      COMPOSE_FILES+=(-f "$candidate/deploy/containers/$overlay")
      break
    fi
  done
done
mapfile -t IMAGE_REFS < <("$COMPOSE" --env-file "$ENV_FILE" "${COMPOSE_FILES[@]}" \
  --profile reid --profile embedding config 2>/dev/null \
  | awk '/^[[:space:]]+image:/ { print $2 }' | sed 's/^"//; s/"$//' | sort -u)
[ "${#IMAGE_REFS[@]}" -gt 0 ] || fail "compose config resolved no images"

echo "Images to save:"
images_bytes=0
for ref in "${IMAGE_REFS[@]}"; do
  docker image inspect "$ref" >/dev/null 2>&1 \
    || fail "image '$ref' is not present locally; deploy (or docker pull) it before bundling"
  size="$(docker image inspect -f '{{.Size}}' "$ref")"
  images_bytes=$((images_bytes + size))
  echo "  $ref ($(awk -v b="$size" 'BEGIN { printf "%.1f GiB", b / 1073741824 }'))"
done

models_mib="$(du -sm "$MODEL_DIR" | awk '{print $1}')"
src_mib="$(du -sm "$RELEASE_DIR" | awk '{print $1}')"
need_mib=$(( (images_bytes / 1048576 + models_mib + src_mib) * 22 / 10 + 2048 ))
OUT="${OUT:-$ROOT}"
mkdir -p "$OUT"
free_mib="$(df -Pm "$OUT" | awk 'NR==2 { print $4 }')"
[ "$free_mib" -ge "$need_mib" ] \
  || fail "only ${free_mib} MiB free in $OUT; the staging copy plus the tar need ~${need_mib} MiB"

NAME="sightindex-offline-$(date +%Y%m%d-%H%M%S)"
STAGE="$OUT/$NAME"
rm -rf "$STAGE"
mkdir -p "$STAGE/images" "$STAGE/models" "$STAGE/src"

echo "Saving images"
for ref in "${IMAGE_REFS[@]}"; do
  # image refs contain '/' and ':'; flatten to a filename
  tar_name="$(printf '%s' "$ref" | tr '/:' '__').tar"
  docker save -o "$STAGE/images/$tar_name" "$ref"
done

echo "Copying models (${models_mib} MiB)"
cp -a "$MODEL_DIR/." "$STAGE/models/"

echo "Copying release source (${src_mib} MiB)"
cp -a "$RELEASE_DIR/." "$STAGE/src/"
# The release dir is a deploy.sh copy; install.sh is the offline entry point.
if [ ! -f "$STAGE/src/deploy/containers/install.sh" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  [ -f "$SCRIPT_DIR/../deploy/containers/install.sh" ] \
    && cp "$SCRIPT_DIR/../deploy/containers/install.sh" "$STAGE/src/deploy/containers/install.sh"
fi

REID_ENABLED="$(env_value REID_ENABLED)"
REID_ENABLED="${REID_ENABLED:-false}"
REID_REVISION="$(env_value REID_CHECKPOINT_REVISION)"

cat > "$STAGE/manifest.env" <<EOF
SIGHTINDEX_IMAGE=$SIGHTINDEX_IMAGE
RELEASE=$(basename "$RELEASE_DIR")
REID_ENABLED=$REID_ENABLED
REID_CHECKPOINT_REVISION=$REID_REVISION
MODELS_MIB=$models_mib
BUNDLE_DATE=$(date +%Y-%m-%dT%H:%M:%S%z)
EOF

echo "Checksumming (this is the slow part for multi-GiB bundles)"
( cd "$STAGE" && find . -type f ! -name checksums.sha256 -exec sha256sum {} + > checksums.sha256 )

echo "Creating $OUT/$NAME.tar"
tar -cf "$OUT/$NAME.tar" -C "$OUT" "$NAME"

# install.sh ships next to the tar; the target machine needs nothing else.
cp "$STAGE/src/deploy/containers/install.sh" "$OUT/$NAME-install.sh" 2>/dev/null || true

bundle_bytes="$(wc -c < "$OUT/$NAME.tar" | tr -d ' ')"
rm -rf "$STAGE"

echo
echo "Bundle ready:"
echo "  $OUT/$NAME.tar ($(awk -v b="$bundle_bytes" 'BEGIN { printf "%.1f GiB", b / 1073741824 }'))"
echo "  $OUT/$NAME-install.sh"
echo "Target machine: bash $NAME-install.sh --offline $NAME.tar [--stacks \"base reid\"]"
if [ "$REID_ENABLED" = "true" ]; then
  echo "Note: reid is enabled; if you swap the sapiensid checkpoint later, update"
  echo "      REID_CHECKPOINT_REVISION (currently $REID_REVISION) and re-check /ready."
fi
