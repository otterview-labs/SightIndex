#!/usr/bin/env bash
# Model-only adapter for the same profile and configuration used by deployment.
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${SIGHTINDEX_ROOT:-/data/sightindex-bj-test}"
TARGET=containers
ENV_FILE=""
STACKS=""
ARGS=()
usage() {
  cat <<'EOF'
Usage: bash deploy.sh --models-only [--target containers|rtx5090]
  --env-file FILE --model-manifest FILE --stacks "base [reid] [embedding] [semantic]"
  --check
or:
  --model-source DIR --acknowledge-model-terms
or:
  --download-models --acknowledge-model-terms
or (read-only interrupted-state diagnosis):
  --diagnose-models
or (maintenance only, no preparation or service start):
  --recover-models --confirm-no-active-preparation [--quarantine-partial ROLE]

This only prepares/checks reviewed model files; it starts no application or model
service. Acknowledgement records review, not permission for commercial use.
EOF
}
while [ "$#" -gt 0 ]; do
  case "$1" in
    --target|--root|--source|--env-file|--stacks|--model-manifest|--model-source|--quarantine-partial)
      [ "$#" -ge 2 ] || { echo 'ERROR: missing model option value' >&2; exit 2; }
      case "$1" in
        --target) TARGET="$2" ;; --root) ROOT="$2" ;; --source) SOURCE="$2" ;;
        --env-file) ENV_FILE="$2" ;; --stacks) STACKS="$2" ;;
        --model-manifest) ARGS+=(--manifest "$2") ;;
        --model-source) ARGS+=(--source-dir "$2") ;;
        --quarantine-partial) ARGS+=(--quarantine-partial "$2") ;;
      esac
      shift 2 ;;
    --download-models) ARGS+=(--download); shift ;;
    --check|--acknowledge-model-terms|--diagnose-models|--recover-models|--confirm-no-active-preparation)
      ARGS+=("$1"); shift ;;
    --prepare-models) shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo 'ERROR: unsupported model-only option' >&2; exit 2 ;;
  esac
done
case "$TARGET" in
  containers) ENV_FILE="${ENV_FILE:-$ROOT/.env}"; STACKS="${STACKS:-base}" ;;
  rtx5090) ENV_FILE="${ENV_FILE:-$SOURCE/.env}"; STACKS="${STACKS:-base reid}" ;;
  *) echo 'ERROR: unknown model deployment target' >&2; exit 2 ;;
esac
exec "${SIGHTINDEX_DEPLOY_PYTHON:-python3}" "$SOURCE/deploy/models/setup.py" \
  --target "$TARGET" --source "$SOURCE" --env-file "$ENV_FILE" \
  --stacks "$STACKS" ${ARGS[@]+"${ARGS[@]}"}
