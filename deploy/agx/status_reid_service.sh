#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

set -a
[ -f .env ] && . ./.env
set +a

REID_SERVICE_PORT="${REID_SERVICE_PORT:-18031}"

echo "reid process:"
if [ -f logs/reid_service.pid ]; then
  ps -fp "$(cat logs/reid_service.pid)" || true
else
  echo "no pid file"
fi

echo "--- reid port:"
ss -lntp | grep ":$REID_SERVICE_PORT" || echo "not listening"

echo "--- reid health:"
curl -fsS "http://127.0.0.1:$REID_SERVICE_PORT/health" || true
echo

echo "--- reid readiness:"
curl -fsS --max-time "${REID_STATUS_TIMEOUT_SECONDS:-5}" \
  "http://127.0.0.1:$REID_SERVICE_PORT/ready" || echo "not ready"
echo
