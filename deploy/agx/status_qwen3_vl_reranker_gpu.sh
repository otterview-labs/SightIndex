#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
set -a
[ -f .env ] && . ./.env
set +a

NAME=${QWEN3_VL_RERANKER_CONTAINER:-sightindex-qwen3-vl-reranker}
docker ps --filter "name=$NAME" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
curl -fsS http://127.0.0.1:${QWEN3_VL_RERANKER_PORT:-18022}/health || true
printf '\n'
