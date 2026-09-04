#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

APP_PORT="${APP_PORT:-8000}"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

echo "process:"
if [ -f logs/uvicorn.pid ]; then
  ps -fp "$(cat logs/uvicorn.pid)" || true
else
  echo "no pid file"
fi

echo "--- port:"
ss -lntp | grep ":$APP_PORT" || echo "not listening"

echo "--- health:"
curl -fsS "http://127.0.0.1:$APP_PORT/health" || true

if [ "${VISUAL_EMBEDDING_PROVIDER:-}" = "qwen3_vl_http" ]; then
  echo
  echo "--- embedding service:"
  bash deploy/agx/status_embedding_service.sh
fi

bash deploy/agx/status_reid_service.sh || true
