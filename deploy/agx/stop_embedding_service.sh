#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

set -a
[ -f .env ] && . ./.env
set +a

EMBEDDING_SERVICE_PORT="${EMBEDDING_SERVICE_PORT:-18021}"

if [ -f logs/embedding_service.pid ]; then
  old_pid="$(cat logs/embedding_service.pid || true)"
  if [ -n "$old_pid" ]; then
    kill "$old_pid" >/dev/null 2>&1 || true
    sleep 1
    kill -9 "$old_pid" >/dev/null 2>&1 || true
  fi
  rm -f logs/embedding_service.pid
fi

port_pids="$(lsof -t -i:"$EMBEDDING_SERVICE_PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$port_pids" ]; then
  kill $port_pids >/dev/null 2>&1 || true
fi
