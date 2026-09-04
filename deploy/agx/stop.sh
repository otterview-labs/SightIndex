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

if [ -f logs/uvicorn.pid ]; then
  old_pid="$(cat logs/uvicorn.pid || true)"
  if [ -n "$old_pid" ]; then
    kill "$old_pid" >/dev/null 2>&1 || true
    sleep 1
    kill -9 "$old_pid" >/dev/null 2>&1 || true
  fi
  rm -f logs/uvicorn.pid
fi

port_pids="$(lsof -t -i:"$APP_PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$port_pids" ]; then
  kill $port_pids >/dev/null 2>&1 || true
fi

if [ "${YOLO_SERVICE_AUTOSTOP:-false}" = "true" ]; then
  bash deploy/agx/stop_yolo_service.sh
fi

if [ "${EMBEDDING_SERVICE_AUTOSTOP:-false}" = "true" ]; then
  bash deploy/agx/stop_embedding_service.sh
fi

bash deploy/agx/stop_reid_service.sh || true
