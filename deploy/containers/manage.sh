#!/usr/bin/env bash
# One-entry manage script for the SightIndex container deployment.
#
# Wraps docker compose with the env file, release directory, overlay compose
# files, and profiles used by the sightindex-bj-test deployment, so operators
# do not hand-assemble --env-file/-f/--profile combinations.
#
# Usage:
#   manage.sh [--env-file FILE] [--release DIR] COMMAND [STACK ...]
#
# Commands:
#   up        Create and start the selected stacks (default action)
#   down      Stop the whole deployment (keeps volumes; never runs down -v)
#   restart   Restart the services of the selected stacks
#   status    Show compose ps plus the API health endpoint
#   logs      Follow logs of the selected stacks (--tail=100)
#
# Stacks (combinable; default: base):
#   base      postgres etcd minio milvus api
#   reid      + reid service (requires REID_ENABLED=true in the env file)
#   embedding + Qwen3-VL embedding service and API visual-embedding wiring
#              (requires the QWEN_* keys in the env file)
#   semantic  + semantic-search API settings overlay
#
# Defaults resolve for /data/sightindex-bj-test; override with SIGHTINDEX_ROOT
# or the flags. Overlay compose files (compose.embedding.yaml,
# compose.semantic-search.yaml) are searched across all releases newest-first,
# so split releases keep working. Release names must not contain spaces.

set -euo pipefail

ROOT="${SIGHTINDEX_ROOT:-/data/sightindex-bj-test}"
ENV_FILE=""
RELEASE=""
COMMAND=""
STACKS=()

usage() {
  sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --env-file)
      [ $# -ge 2 ] || usage
      ENV_FILE="$2"; shift 2 ;;
    --release)
      [ $# -ge 2 ] || usage
      RELEASE="$2"; shift 2 ;;
    -h|--help)
      usage ;;
    up|down|restart|status|logs)
      [ -z "$COMMAND" ] || { echo "Only one command allowed" >&2; exit 2; }
      COMMAND="$1"; shift ;;
    base|reid|embedding|semantic)
      STACKS+=("$1"); shift ;;
    *)
      echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[ -n "$COMMAND" ] || usage
if [ "${#STACKS[@]}" -eq 0 ]; then
  STACKS=(base)
fi

ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "Env file not found: $ENV_FILE" >&2
  exit 1
fi

env_value() {
  # Print the value of KEY from the env file; empty when unset.
  awk -F= -v key="$1" '$1 == key { sub(/^[^=]*=/, ""); print; exit }' "$ENV_FILE"
}

base_compose_in() {
  if [ -f "$1/deploy/containers/compose.yaml" ]; then
    echo "$1/deploy/containers/compose.yaml"
  elif [ -f "$1/deploy/containers/compose.base.yaml" ]; then
    echo "$1/deploy/containers/compose.base.yaml"
  fi
}

# Resolve the release directory: explicit flag, else the newest release that
# ships a base compose file.
if [ -z "$RELEASE" ]; then
  for candidate in $(ls -1dt "$ROOT"/releases/*/ 2>/dev/null || true); do
    candidate="${candidate%/}"
    if [ -n "$(base_compose_in "$candidate")" ]; then
      RELEASE="$candidate"
      break
    fi
  done
fi
if [ -z "$RELEASE" ]; then
  echo "No release with a base compose file under $ROOT/releases" >&2
  exit 1
fi
BASE_COMPOSE="$(base_compose_in "$RELEASE")"
if [ -z "$BASE_COMPOSE" ]; then
  echo "No compose.yaml/compose.base.yaml in $RELEASE" >&2
  exit 1
fi

# Overlay files may live in a different release than the base; newest wins.
find_overlay() {
  local name="$1" candidate
  for candidate in "$RELEASE" $(ls -1dt "$ROOT"/releases/*/ 2>/dev/null | sed 's:/$::'); do
    if [ -f "$candidate/deploy/containers/$name" ]; then
      echo "$candidate/deploy/containers/$name"
      return 0
    fi
  done
  return 1
}

COMPOSE_FILES=(-f "$BASE_COMPOSE")
PROFILES=()
SERVICES=(postgres etcd minio milvus api)

for stack in ${STACKS[@]+"${STACKS[@]}"}; do
  case "$stack" in
    base)
      ;;
    reid)
      if [ "$(env_value REID_ENABLED)" != "true" ]; then
        echo "reid stack requested but REID_ENABLED is not 'true' in $ENV_FILE." >&2
        echo "Set REID_ENABLED=true (and check GPU capacity) before enabling it." >&2
        exit 1
      fi
      PROFILES+=(--profile reid)
      SERVICES+=(reid) ;;
    embedding)
      overlay="$(find_overlay compose.embedding.yaml)" || {
        echo "compose.embedding.yaml not found in any release" >&2; exit 1;
      }
      COMPOSE_FILES+=(-f "$overlay")
      for key in QWEN_EMBEDDING_IMAGE QWEN_EMBEDDING_MODEL_DIR QWEN_EMBEDDING_API_KEY; do
        if [ -z "$(env_value "$key")" ]; then
          echo "$key missing in $ENV_FILE; the embedding stack needs the QWEN_* keys" >&2
          for candidate in "$ROOT"/.env*; do
            [ -f "$candidate" ] || continue
            if [ -n "$(awk -F= -v k="$key" '$1 == k { sub(/^[^=]*=/, ""); print; exit }' "$candidate")" ]; then
              echo "  hint: $key is set in $candidate -- pass --env-file or merge" >&2
            fi
          done
          exit 1
        fi
      done
      PROFILES+=(--profile embedding)
      SERVICES+=(embedding) ;;
    semantic)
      overlay="$(find_overlay compose.semantic-search.yaml)" || {
        echo "compose.semantic-search.yaml not found in any release" >&2; exit 1;
      }
      COMPOSE_FILES+=(-f "$overlay") ;;
  esac
done

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
else
  COMPOSE=(docker-compose)
fi

compose() {
  "${COMPOSE[@]}" --env-file "$ENV_FILE" \
    ${COMPOSE_FILES[@]+"${COMPOSE_FILES[@]}"} \
    ${PROFILES[@]+"${PROFILES[@]}"} "$@"
}

echo "Env file:    $ENV_FILE"
echo "Release:     $RELEASE"
echo "Base:        $BASE_COMPOSE"

# The running compose project may have been started from an older release;
# warn when this command would manage (or re-point) it to a different base.
project="$(sed -n 's/^name:[[:space:]]*//p' "$BASE_COMPOSE" | head -1)"
if [ -n "$project" ]; then
  running_base="$("${COMPOSE[@]}" ls 2>/dev/null | awk -v p="$project" '$1 == p {print $NF}' | head -1)"
  if [ -n "$running_base" ] && [ "$running_base" != "$BASE_COMPOSE" ]; then
    echo "WARNING: the running '$project' project was started from:" >&2
    echo "         $running_base" >&2
    echo "         this command targets $BASE_COMPOSE instead;" >&2
    echo "         'up'/'restart' will re-create containers from it." >&2
  fi
fi

if [ "${#PROFILES[@]}" -gt 0 ]; then
  echo "Profiles:    ${PROFILES[*]}"
fi
if [ "$COMMAND" = "up" ] || [ "$COMMAND" = "restart" ]; then
  echo "Services:    ${SERVICES[*]}"
fi

case "$COMMAND" in
  up)
    compose up -d ${SERVICES[@]+"${SERVICES[@]}"}
    ;;
  restart)
    compose restart ${SERVICES[@]+"${SERVICES[@]}"}
    ;;
  down)
    # Tear down the whole project, profile services included. Volumes and data
    # are intentionally preserved; never use down -v here. Include the overlay
    # compose files when present, otherwise services defined only in an overlay
    # (e.g. embedding) would be orphaned by a base-only down.
    down_files=(-f "$BASE_COMPOSE")
    for overlay_name in compose.embedding.yaml compose.semantic-search.yaml; do
      if overlay="$(find_overlay "$overlay_name")"; then
        down_files+=(-f "$overlay")
      fi
    done
    "${COMPOSE[@]}" --env-file "$ENV_FILE" "${down_files[@]}" \
      --profile reid --profile embedding down
    echo "Volumes kept. Data lives in named volumes and $ROOT/{media,models,cache}."
    ;;
  logs)
    compose logs --tail=100 -f ${SERVICES[@]+"${SERVICES[@]}"}
    ;;
  status)
    compose ps
    bind="$(env_value API_BIND)"; port="$(env_value API_PORT)"
    bind="${bind:-127.0.0.1}"; port="${port:-18030}"
    echo
    echo "API health: http://$bind:$port/health"
    if curl -fsS -m 5 "http://$bind:$port/health"; then
      echo
    else
      echo "(API not answering on http://$bind:$port/health -- it may be stopped)"
      exit 1
    fi
    ;;
esac
