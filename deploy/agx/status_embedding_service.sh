#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

set -a
[ -f .env ] && . ./.env
set +a

EMBEDDING_SERVICE_PORT="${EMBEDDING_SERVICE_PORT:-18021}"

echo "embedding process:"
if [ -f logs/embedding_service.pid ]; then
  ps -fp "$(cat logs/embedding_service.pid)" || true
else
  echo "no pid file"
fi

echo "--- embedding port:"
ss -lntp | grep ":$EMBEDDING_SERVICE_PORT" || echo "not listening"

echo "--- embedding health:"
curl -fsS "http://127.0.0.1:$EMBEDDING_SERVICE_PORT/health" || true
