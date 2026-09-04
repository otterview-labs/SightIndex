#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

set -a
[ -f .env ] && . ./.env
set +a

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

EMBEDDING_SERVICE_HOST="${EMBEDDING_SERVICE_HOST:-127.0.0.1}"
EMBEDDING_SERVICE_PORT="${EMBEDDING_SERVICE_PORT:-18021}"
EMBEDDING_SERVICE_PROVIDER="${EMBEDDING_SERVICE_PROVIDER:-qwen3_vl}"
RERANK_SERVICE_PROVIDER="${RERANK_SERVICE_PROVIDER:-${VLM_RERANK_PROVIDER:-none}}"
EMBEDDING_SERVICE_PYTHONPATH="${QWEN3_VL_EMBEDDING_PYTHONPATH:-}"
EMBEDDING_SERVICE_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ -n "$EMBEDDING_SERVICE_PYTHONPATH" ]; then
  IFS=':' read -r -a embedding_pythonpath_entries <<< "$EMBEDDING_SERVICE_PYTHONPATH"
  for embedding_pythonpath_entry in "${embedding_pythonpath_entries[@]}"; do
    torch_lib_dir="$embedding_pythonpath_entry/torch/lib"
    if [ -d "$torch_lib_dir" ]; then
      if [ -n "$EMBEDDING_SERVICE_LD_LIBRARY_PATH" ]; then
        EMBEDDING_SERVICE_LD_LIBRARY_PATH="$torch_lib_dir:$EMBEDDING_SERVICE_LD_LIBRARY_PATH"
      else
        EMBEDDING_SERVICE_LD_LIBRARY_PATH="$torch_lib_dir"
      fi
    fi
  done
fi

if [ -f logs/embedding_service.pid ]; then
  old_pid="$(cat logs/embedding_service.pid || true)"
  if [ -n "$old_pid" ]; then
    kill "$old_pid" >/dev/null 2>&1 || true
  fi
fi

port_pids="$(lsof -t -i:"$EMBEDDING_SERVICE_PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$port_pids" ]; then
  kill $port_pids >/dev/null 2>&1 || true
fi

nohup env \
  PYTHONPATH="$EMBEDDING_SERVICE_PYTHONPATH" \
  LD_LIBRARY_PATH="$EMBEDDING_SERVICE_LD_LIBRARY_PATH" \
  APP_NAME="${APP_NAME:-SightIndex} Embedding" \
  AUTO_CREATE_TABLES=false \
  STREAM_AUTOSTART_RUNNING=false \
  VISUAL_EMBEDDING_PROVIDER="$EMBEDDING_SERVICE_PROVIDER" \
  VISUAL_EMBEDDING_SERVICE_URL= \
  VLM_RERANK_PROVIDER="$RERANK_SERVICE_PROVIDER" \
  VLM_RERANK_MODEL="${VLM_RERANK_MODEL:-Qwen/Qwen3-VL-Reranker-2B}" \
  .venv/bin/python -m uvicorn main:app \
    --host "$EMBEDDING_SERVICE_HOST" \
    --port "$EMBEDDING_SERVICE_PORT" \
  > logs/embedding_service.log 2>&1 < /dev/null &
echo $! > logs/embedding_service.pid
