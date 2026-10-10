#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs data/uploads data/videos data/crops data/thumbnails data/frames

set -a
[ -f .env ] && . ./.env
set +a

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.agx.txt}"
.venv/bin/python -m pip install -U pip > logs/pip_bootstrap.log 2>&1
.venv/bin/python -m pip install -r "$REQUIREMENTS_FILE" > logs/pip_install.log 2>&1

# The console is a Vue SPA built here rather than committed, so the app has a bundle to serve.
# Set FRONTEND_BUILD=skip to reuse an existing frontend/dist and avoid the npm round trip.
if [ "${FRONTEND_BUILD:-auto}" != "skip" ]; then
  # nvm is normally wired into ~/.bashrc, which only interactive shells read. A script, cron
  # job or non-interactive ssh command therefore sees the distro node, which is often far too
  # old for the Vite toolchain; prefer the newest nvm runtime before giving up.
  node_major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
  if [ "${node_major:-0}" -lt 20 ] && [ -d "${NVM_DIR:-$HOME/.nvm}/versions/node" ]; then
    nvm_node="$(ls -1 "${NVM_DIR:-$HOME/.nvm}/versions/node" 2>/dev/null | sort -V | tail -1)"
    if [ -n "$nvm_node" ] && [ -x "${NVM_DIR:-$HOME/.nvm}/versions/node/$nvm_node/bin/npm" ]; then
      export PATH="${NVM_DIR:-$HOME/.nvm}/versions/node/$nvm_node/bin:$PATH"
      node_major="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
      echo "start.sh: using nvm node $nvm_node for the frontend build" >&2
    fi
  fi
  if ! command -v npm > /dev/null 2>&1; then
    echo "npm not found; install Node.js or set FRONTEND_BUILD=skip with a prebuilt frontend/dist" >&2
    exit 1
  fi
  if [ "${node_major:-0}" -lt 20 ]; then
    echo "Node.js ${node_major}.x is too old for the frontend toolchain (need 20+)." >&2
    echo "Install a newer Node.js and put it on PATH, or set FRONTEND_BUILD=skip with a prebuilt frontend/dist." >&2
    exit 1
  fi
  ( cd frontend && npm ci && npm run build ) > logs/frontend_build.log 2>&1
fi

if [ ! -f frontend/dist/index.html ]; then
  echo "frontend/dist/index.html missing; see logs/frontend_build.log" >&2
  exit 1
fi

if [ "${PERSON_DETECTOR:-}" = "yolo_service" ] && [ "${YOLO_SERVICE_AUTOSTART:-true}" != "false" ]; then
  bash deploy/agx/start_yolo_service.sh > logs/yolo_service.log 2>&1
fi

if [ "${VISUAL_EMBEDDING_PROVIDER:-}" = "qwen3_vl_http" ] && [ "${EMBEDDING_SERVICE_AUTOSTART:-true}" != "false" ]; then
  bash deploy/agx/start_embedding_service.sh > logs/embedding_service_start.log 2>&1
fi

if [ "${REID_ENABLED:-false}" = "true" ] && [ "${REID_SERVICE_AUTOSTART:-true}" != "false" ]; then
  bash deploy/agx/start_reid_service.sh > logs/reid_service_start.log 2>&1
fi

if [ -f logs/uvicorn.pid ]; then
  old_pid="$(cat logs/uvicorn.pid || true)"
  if [ -n "$old_pid" ]; then
    kill "$old_pid" >/dev/null 2>&1 || true
  fi
fi

APP_PORT="${APP_PORT:-8000}"
APP_HOST="${APP_HOST:-127.0.0.1}"
port_pids="$(lsof -t -i:"$APP_PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$port_pids" ]; then
  kill $port_pids >/dev/null 2>&1 || true
fi

nohup .venv/bin/python -m uvicorn main:app --host "$APP_HOST" --port "$APP_PORT" \
  > logs/uvicorn.log 2>&1 < /dev/null &
echo $! > logs/uvicorn.pid
