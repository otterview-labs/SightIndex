#!/usr/bin/env bash
# Stable public entry point; each deployment profile keeps its own safety policy.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=containers
MODELS_ONLY=0
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --target)
      [ "$#" -ge 2 ] || { echo 'ERROR: --target requires containers or rtx5090' >&2; exit 2; }
      TARGET="$2"; shift 2 ;;
    --models-only) MODELS_ONLY=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [ "$MODELS_ONLY" = 1 ]; then
  exec bash "$ROOT_DIR/deploy/models/manage.sh" --target "$TARGET" ${ARGS[@]+"${ARGS[@]}"}
fi
case "$TARGET" in
  containers) exec bash "$ROOT_DIR/deploy/containers/deploy.sh" ${ARGS[@]+"${ARGS[@]}"} ;;
  rtx5090) exec bash "$ROOT_DIR/deploy/rtx5090/install_or_update.sh" ${ARGS[@]+"${ARGS[@]}"} ;;
  *) echo 'ERROR: unknown target; use containers or rtx5090' >&2; exit 2 ;;
esac
