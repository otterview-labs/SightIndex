#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
set -a
[ -f .env ] && . ./.env
set +a

docker rm -f "${QWEN3_VL_RERANKER_CONTAINER:-sightindex-qwen3-vl-reranker}" >/dev/null 2>&1 || true
