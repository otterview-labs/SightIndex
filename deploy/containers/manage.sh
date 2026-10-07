#!/usr/bin/env bash
# One-entry manage script for the SightIndex container deployment.
#
# Wraps docker compose with the env file, release directory, overlay compose
# files, and profiles used by the sightindex-bj-test deployment, so operators
# do not hand-assemble --env-file/-f/--profile combinations.
#
# Usage:
#   manage.sh [--root DIR] [--project-name NAME] [--env-file FILE] [--release DIR] COMMAND [STACK ...]
#   manage.sh [--env-file FILE] [--release DIR] logs [--service SERVICE | api]
#   manage.sh [--env-file FILE] [--release DIR] verify [STACK ...] [--model-smoke]
#
# Commands:
#   up        Create and start the selected stacks
#   down      Stop the whole deployment (keeps volumes; never runs down -v)
#   restart   Restart existing services (does not apply changed image/config)
#   status    Show compose ps plus the API health endpoint
#   logs      Follow logs of the selected stacks (--tail=100)
#   config    Validate selected compose configuration without printing secrets
#   verify    Check selected API/model service images and authenticated acceptance checks
#             Includes an isolated upload smoke; --model-smoke also loads CPU models
#
# Stacks (combinable; default: base):
#   base      postgres etcd minio milvus api
#   reid      + reid service (requires REID_ENABLED=true in the env file)
#   embedding + Qwen3-VL embedding service and API visual-embedding wiring
#              (requires the QWEN_* keys in the env file)
#   semantic  + semantic-search API settings overlay
#
# Defaults resolve for /data/sightindex-bj-test; override with SIGHTINDEX_ROOT
# or the flags. Release defaults to active-release after successful acceptance,
# otherwise the newest non-pending release with a base compose file.
# Overlay compose files (compose.embedding.yaml,
# compose.semantic-search.yaml) are searched across all releases newest-first,
# so split releases keep working. Release names must not contain spaces.

set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_ROOT="/data/sightindex-bj-test"
# Installed ROOT/manage.sh is self-scoped; the source-tree script retains its
# backward-compatible default. An explicit environment or --root may override it.
if [ -d "$SCRIPT_DIRECTORY/releases" ] || [ -f "$SCRIPT_DIRECTORY/project-name" ]; then
  DEFAULT_ROOT="$SCRIPT_DIRECTORY"
fi
ROOT="${SIGHTINDEX_ROOT:-$DEFAULT_ROOT}"
ENV_FILE=""
RELEASE=""
COMMAND=""
STACKS=()
LOG_SERVICES=()
MODEL_SMOKE=0
PROJECT_NAME=""

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

log_service() {
  case "$1" in
    postgres|etcd|minio|milvus|api|reid|embedding)
      LOG_SERVICES+=("$1") ;;
    *)
      echo "Unknown service: $1 (allowed: postgres etcd minio milvus api reid embedding)" >&2
      exit 2 ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --env-file)
      [ $# -ge 2 ] || usage
      ENV_FILE="$2"; shift 2 ;;
    --root)
      [ $# -ge 2 ] || usage
      ROOT="$2"; shift 2 ;;
    --release)
      [ $# -ge 2 ] || usage
      RELEASE="$2"; shift 2 ;;
    --project-name)
      [ $# -ge 2 ] || usage
      PROJECT_NAME="$2"; shift 2 ;;
    --service)
      [ $# -ge 2 ] || usage
      log_service "$2"; shift 2 ;;
    --model-smoke)
      MODEL_SMOKE=1; shift ;;
    -h|--help)
      usage ;;
    up|down|restart|status|logs|config|verify)
      [ -z "$COMMAND" ] || { echo "Only one command allowed" >&2; exit 2; }
      COMMAND="$1"; shift ;;
    base|reid|embedding|semantic)
      STACKS+=("$1"); shift ;;
    postgres|etcd|minio|milvus|api)
      log_service "$1"; shift ;;
    *)
      echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[ -n "$COMMAND" ] || usage
if [ "${#LOG_SERVICES[@]}" -gt 0 ] && [ "$COMMAND" != "logs" ]; then
  echo "Service selection is only supported by logs." >&2
  exit 2
fi
if [ "$MODEL_SMOKE" = 1 ] && [ "$COMMAND" != "verify" ]; then
  echo "--model-smoke is only supported by verify." >&2
  exit 2
fi
if [ "${#STACKS[@]}" -eq 0 ]; then
  STACKS=(base)
fi

ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "Env file not found: $ENV_FILE" >&2
  exit 1
fi

env_value() {
  # Read scalar deployment settings without sourcing/evaluating the env file.
  # Accept quoted values and comments as Compose does; the last occurrence wins.
  awk -v key="$1" '
    {
      line = $0
      sub(/\r$/, "", line)
      sub(/^[[:space:]]*export[[:space:]]+/, "", line)
      separator = index(line, "=")
      if (!separator) next
      name = substr(line, 1, separator - 1)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
      if (name != key) next
      value = substr(line, separator + 1)
      sub(/^[[:space:]]+/, "", value)
      quote = substr(value, 1, 1)
      if (quote == "\"" || quote == sprintf("%c", 39)) {
        closing = index(substr(value, 2), quote)
        if (closing) value = substr(value, 2, closing - 1)
      } else {
        sub(/[[:space:]]+#.*$/, "", value)
        sub(/[[:space:]]+$/, "", value)
      }
      resolved = value
      found = 1
    }
    END { if (found) print resolved }
  ' "$ENV_FILE"
}

configured_project="$(env_value COMPOSE_PROJECT_NAME)"
accepted_project=""
if [ -e "$ROOT/project-name" ] || [ -L "$ROOT/project-name" ]; then
  if [ -L "$ROOT/project-name" ] || [ ! -f "$ROOT/project-name" ]; then
    echo "Invalid project-name marker." >&2; exit 1
  fi
  accepted_project="$(<"$ROOT/project-name")"
  if ! [[ "$accepted_project" =~ ^[a-z0-9][a-z0-9_-]{0,62}$ ]]; then
    echo "Invalid accepted project name." >&2; exit 1
  fi
fi
PROJECT_NAME="${PROJECT_NAME:-${configured_project:-${accepted_project:-sightindex-bj-test}}}"
if ! [[ "$PROJECT_NAME" =~ ^[a-z0-9][a-z0-9_-]{0,62}$ ]]; then
  echo "Invalid Compose project name." >&2; exit 2
fi
if { [ -n "$configured_project" ] && [ "$PROJECT_NAME" != "$configured_project" ]; } \
  || { [ -n "$accepted_project" ] && [ "$PROJECT_NAME" != "$accepted_project" ]; }; then
  echo "Project name differs from configuration or accepted deployment root." >&2; exit 1
fi

base_compose_in() {
  if [ -f "$1/deploy/containers/compose.yaml" ]; then
    echo "$1/deploy/containers/compose.yaml"
  elif [ -f "$1/deploy/containers/compose.base.yaml" ]; then
    echo "$1/deploy/containers/compose.base.yaml"
  fi
}

# An explicit release wins. A successful deployment publishes active-release;
# a newer staged directory must not silently take over everyday operations.
if [ -z "$RELEASE" ] && { [ -e "$ROOT/active-release" ] || [ -L "$ROOT/active-release" ]; }; then
  if [ -L "$ROOT/active-release" ] || [ ! -f "$ROOT/active-release" ]; then
    echo "Invalid active-release marker: expected a regular file." >&2
    exit 1
  fi
  RELEASE="$(<"$ROOT/active-release")"
  case "$RELEASE" in
    ''|*$'\n'*|*$'\r'*)
      echo "Invalid active-release marker: expected one absolute release path." >&2
      exit 1 ;;
    /*)
      ;;
    *)
      echo "Invalid active-release marker: release path must be absolute." >&2
      exit 1 ;;
  esac
  if [ ! -d "$ROOT/releases" ] || [ ! -d "$RELEASE" ]; then
    echo "Invalid active-release marker: release directory does not exist." >&2
    exit 1
  fi
  release_root="$(cd "$ROOT/releases" && pwd -P)"
  canonical_release="$(cd "$RELEASE" && pwd -P)"
  case "$canonical_release" in
    "$release_root"/*)
      RELEASE="$canonical_release" ;;
    *)
      echo "Invalid active-release marker: release must be inside $ROOT/releases." >&2
      exit 1 ;;
  esac
fi

# Legacy installations without an acceptance marker retain newest-first lookup.
if [ -z "$RELEASE" ]; then
  for candidate in $(ls -1dt "$ROOT"/releases/*/ 2>/dev/null || true); do
    candidate="${candidate%/}"
    [ ! -f "$candidate/.deployment-pending" ] || continue
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
# Pending candidates cannot supply an overlay for another accepted release.
find_overlay() {
  local name="$1" candidate
  for candidate in "$RELEASE" $(ls -1dt "$ROOT"/releases/*/ 2>/dev/null | sed 's:/$::'); do
    if [ "$candidate" != "$RELEASE" ] && [ -f "$candidate/.deployment-pending" ]; then
      continue
    fi
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
  "${COMPOSE[@]}" --project-name "$PROJECT_NAME" --env-file "$ENV_FILE" \
    ${COMPOSE_FILES[@]+"${COMPOSE_FILES[@]}"} \
    ${PROFILES[@]+"${PROFILES[@]}"} "$@"
}

verify_service_image() {
  # Validate actual image content, not only mutable tags or healthy responses.
  local service="$1" image_key="$2" label="$3"
  local container configured_image expected_id running_id
  container="$(compose ps --quiet "$service")"
  case "$container" in
    ''|*[!a-fA-F0-9]*)
      echo "Expected one running $label container in the selected compose project." >&2
      return 1 ;;
  esac
  configured_image="$(env_value "$image_key")"
  [ -n "$configured_image" ] || {
    echo "$image_key is missing in $ENV_FILE." >&2
    return 1
  }
  expected_id="$(docker image inspect --format '{{.Id}}' "$configured_image")"
  running_id="$(docker inspect --format '{{.Image}}' "$container")"
  if [ -z "$expected_id" ] || [ "$running_id" != "$expected_id" ]; then
    echo "$label image mismatch: running image content does not match configuration." >&2
    echo "Apply the selected image with 'up' before verifying this release." >&2
    return 1
  fi
  echo "$label image verified: $running_id"
}

echo "Env file:    $ENV_FILE"
echo "Release:     $RELEASE"
echo "Base:        $BASE_COMPOSE"
echo "Project:     $PROJECT_NAME"

# The running compose project may have been started from an older release;
# warn when this command would manage (or re-point) it to a different base.
project="$PROJECT_NAME"
if [ -n "$project" ]; then
  running_base="$("${COMPOSE[@]}" ls 2>/dev/null | awk -v p="$project" '$1 == p {print $NF}' | head -1)"
  if [ -n "$running_base" ] && [ "$running_base" != "$BASE_COMPOSE" ]; then
    echo "WARNING: the running '$project' project was started from:" >&2
    echo "         $running_base" >&2
    echo "         this command targets $BASE_COMPOSE instead;" >&2
    echo "         'up' applies the selected image/config and may recreate containers." >&2
    echo "         'restart' only restarts existing containers; it does not apply image/config changes." >&2
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
    "${COMPOSE[@]}" --project-name "$PROJECT_NAME" --env-file "$ENV_FILE" "${down_files[@]}" \
      --profile reid --profile embedding down
    echo "Volumes kept. Data lives in named volumes and $ROOT/{media,models,cache}."
    ;;
  logs)
    if [ "${#LOG_SERVICES[@]}" -gt 0 ]; then
      compose logs --tail=100 -f "${LOG_SERVICES[@]}"
    else
      compose logs --tail=100 -f ${SERVICES[@]+"${SERVICES[@]}"}
    fi
    ;;
  config)
    compose config --quiet
    ;;
  verify)
    # A mutable tag can point to a newer image while an older container still
    # answers /health. Inspect the selected compose service and compare image IDs
    # before running the acceptance helper inside that exact deployment.
    verify_service_image api SIGHTINDEX_IMAGE API
    checked_reid=0
    checked_embedding=0
    for stack in ${STACKS[@]+"${STACKS[@]}"}; do
      case "$stack" in
        reid)
          if [ "$checked_reid" = 0 ]; then
            verify_service_image reid SIGHTINDEX_IMAGE ReID
            checked_reid=1
          fi ;;
        embedding)
          if [ "$checked_embedding" = 0 ]; then
            verify_service_image embedding QWEN_EMBEDDING_IMAGE Embedding
            checked_embedding=1
          fi ;;
      esac
    done
    verify_flags=(--stacks "${STACKS[*]}" --upload-smoke)
    if [ "$MODEL_SMOKE" = 1 ]; then
      verify_flags+=(--model-smoke)
    fi
    compose exec -T api python /opt/sightindex/deployment_verify.py "${verify_flags[@]}"
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
