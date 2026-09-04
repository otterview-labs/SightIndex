#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

set -a
[ -f .env ] && . ./.env
set +a

CONTAINER_NAME="${YOLO_SERVICE_CONTAINER:-sightindex-yolo-person}"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
