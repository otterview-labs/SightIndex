#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

set -a
[ -f .env ] && . ./.env
set +a

IMAGE="${QWEN3_VL_RERANKER_IMAGE:-}"
NAME="${QWEN3_VL_RERANKER_CONTAINER:-sightindex-qwen3-vl-reranker}"
PORT="${QWEN3_VL_RERANKER_PORT:-18022}"
MODEL="${VLM_RERANK_MODEL:-}"
KEY="${VLM_RERANK_SERVICE_API_KEY:-}"

if [ -z "$IMAGE" ] || [ -z "$MODEL" ]; then
  echo "QWEN3_VL_RERANKER_IMAGE and VLM_RERANK_MODEL are required" >&2
  exit 1
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --runtime nvidia \
  --network host \
  -e SIGHTINDEX_ROOT=/workspace \
  -e VLM_RERANK_MODEL="$MODEL" \
  -e VLM_RERANK_SERVICE_API_KEY="$KEY" \
  -v "$ROOT_DIR:/workspace:ro" \
  -w /workspace \
  "$IMAGE" \
  python3 -m uvicorn deploy.agx.reranker_service.server:app \
    --host 127.0.0.1 \
    --port "$PORT" >/dev/null
