#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

set -a
[ -f .env ] && . ./.env
set +a

CONTAINER_NAME="${YOLO_SERVICE_CONTAINER:-sightindex-yolo-person}"
IMAGE="${YOLO_SERVICE_IMAGE:-}"
PORT="${YOLO_SERVICE_PORT:-19121}"
HOST="${YOLO_SERVICE_HOST:-127.0.0.1}"
DEVICE="${YOLO_SERVICE_DEVICE:-cuda:0}"
MODEL_PATH="${YOLO_SERVICE_MODEL_PATH:-}"
WORKSPACE="${YOLO_SERVICE_WORKSPACE:-}"

if [ -z "$IMAGE" ] || [ -z "$MODEL_PATH" ] || [ -z "$WORKSPACE" ]; then
  echo "YOLO_SERVICE_IMAGE, YOLO_SERVICE_MODEL_PATH, and YOLO_SERVICE_WORKSPACE are required" >&2
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "YOLO model not found: $MODEL_PATH" >&2
  exit 1
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$CONTAINER_NAME" \
  --network host \
  --runtime nvidia \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$WORKSPACE:/workspace:ro" \
  -v "$ROOT_DIR:$ROOT_DIR:ro" \
  "$IMAGE" \
  python3 -u /workspace/scripts/yolo_temp_service.py \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --model "$MODEL_PATH" >/dev/null

for _ in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "YOLO service started on port $PORT with model $MODEL_PATH"
    exit 0
  fi
  sleep 1
done

echo "YOLO service did not become healthy on port $PORT" >&2
docker logs --tail 80 "$CONTAINER_NAME" >&2 || true
exit 1
