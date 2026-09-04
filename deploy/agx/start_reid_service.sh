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

REID_SERVICE_HOST="${REID_SERVICE_HOST:-127.0.0.1}"
REID_SERVICE_PORT="${REID_SERVICE_PORT:-18031}"
REID_CHECKPOINT_DIR="${REID_CHECKPOINT_DIR:-$ROOT_DIR/data/models/sapiensid_wb12m}"
# systemd starts services with no HOME, and set -u turns that into an instant exit -- the unit
# restart-looped 17 times before anything said why. Name the path outright.
REID_POSE_WEIGHTS="${REID_POSE_WEIGHTS:-${HOME:-/root}/.cache/yolov8n-pose.pt}"

REID_VENDOR_SRC="$ROOT_DIR/deploy/agx/reid_service/sapiensid/tasks/sapiensID/src"
required_reid_assets=(
  "$REID_CHECKPOINT_DIR/model.pth"
  "$REID_CHECKPOINT_DIR/model.yaml"
  "$REID_VENDOR_SRC/aligners/configs/yolo_dfa.yaml"
  "$REID_VENDOR_SRC/aligners/keypoint_predictor/pretrained_models/aligners/dfa_mobilenetv4_medium/mobilenetv4_Final.pth"
  "$REID_POSE_WEIGHTS"
)
for required_reid_asset in "${required_reid_assets[@]}"; do
  if [ ! -f "$required_reid_asset" ]; then
    echo "ReID inference asset missing: $required_reid_asset" >&2
    echo "See deploy/agx/reid_service/README.md for the complete inference manifest." >&2
    exit 1
  fi
done

# The Jetson torch stack lives in its own venv; expose it the way the embedding service does.
REID_PYTHONPATH="${REID_SERVICE_PYTHONPATH:-${QWEN3_VL_EMBEDDING_PYTHONPATH:-}}"
REID_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ -n "$REID_PYTHONPATH" ]; then
  IFS=':' read -r -a reid_pythonpath_entries <<< "$REID_PYTHONPATH"
  for reid_pythonpath_entry in "${reid_pythonpath_entries[@]}"; do
    torch_lib_dir="$reid_pythonpath_entry/torch/lib"
    if [ -d "$torch_lib_dir" ]; then
      if [ -n "$REID_LD_LIBRARY_PATH" ]; then
        REID_LD_LIBRARY_PATH="$torch_lib_dir:$REID_LD_LIBRARY_PATH"
      else
        REID_LD_LIBRARY_PATH="$torch_lib_dir"
      fi
    fi
  done
fi

if [ -f logs/reid_service.pid ]; then
  old_pid="$(cat logs/reid_service.pid || true)"
  if [ -n "$old_pid" ]; then
    kill "$old_pid" >/dev/null 2>&1 || true
  fi
fi

port_pids="$(lsof -t -i:"$REID_SERVICE_PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$port_pids" ]; then
  kill $port_pids >/dev/null 2>&1 || true
fi

# Under a supervisor the process must stay in the foreground, or the supervisor watches a shell
# that exits immediately and never notices the service dying. Everything above -- asset checks,
# torch library paths -- is wanted either way, so the two modes share it.
if [ "${REID_SERVICE_FOREGROUND:-0}" = "1" ]; then
  exec env \
    PYTHONPATH="$REID_PYTHONPATH" \
    LD_LIBRARY_PATH="$REID_LD_LIBRARY_PATH" \
    SIGHTINDEX_ROOT="$ROOT_DIR" \
    REID_CHECKPOINT_DIR="$REID_CHECKPOINT_DIR" \
    REID_DEVICE="${REID_DEVICE:-}" \
    REID_SERVICE_API_KEY="${REID_SERVICE_API_KEY:-}" \
    .venv/bin/python -m uvicorn server:app \
      --app-dir deploy/agx/reid_service \
      --host "$REID_SERVICE_HOST" \
      --port "$REID_SERVICE_PORT"
fi

nohup env \
  PYTHONPATH="$REID_PYTHONPATH" \
  LD_LIBRARY_PATH="$REID_LD_LIBRARY_PATH" \
  SIGHTINDEX_ROOT="$ROOT_DIR" \
  REID_CHECKPOINT_DIR="$REID_CHECKPOINT_DIR" \
  REID_DEVICE="${REID_DEVICE:-}" \
  REID_SERVICE_API_KEY="${REID_SERVICE_API_KEY:-}" \
  .venv/bin/python -m uvicorn server:app \
    --app-dir deploy/agx/reid_service \
    --host "$REID_SERVICE_HOST" \
    --port "$REID_SERVICE_PORT" \
  > logs/reid_service.log 2>&1 < /dev/null &
echo $! > logs/reid_service.pid
