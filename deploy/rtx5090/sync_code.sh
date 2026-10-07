#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
skip_frontend=false
if [ "${1:-}" = "--skip-frontend" ]; then
  skip_frontend=true
  shift
fi

target="${1:-}"
remote_dir="${2:-/opt/sightindex}"
ssh_port="${SSH_PORT:-22}"

if [ -z "$target" ]; then
  printf 'usage: %s [--skip-frontend] user@host [remote-dir]\n' "$0" >&2
  exit 2
fi
case "$remote_dir" in
  ""|/|/home|/root)
    printf 'refusing unsafe remote directory: %s\n' "$remote_dir" >&2
    exit 2
    ;;
esac

if [ "$skip_frontend" = false ]; then
  (cd "$ROOT_DIR/frontend" && npm ci && npm run build && npm run test:reid \
    && npm run test:search && npm run test:observations && npm run test:playback)
fi

# Do not rely on .gitignore here. Runtime state includes a legacy root-level sightindex.db, and a
# normal rsync will otherwise overwrite it before the deployment script gets a chance to back it
# up. The transfer is intentionally non-deleting so models and site-specific files are preserved.
rsync -az --progress \
  -e "ssh -p $ssh_port -o StrictHostKeyChecking=yes" \
  --exclude '/.git/' \
  --exclude '/.agents/' \
  --exclude '/.codex/' \
  --exclude '.DS_Store' \
  --exclude '/:memory:.ses' \
  --exclude '/.env' \
  --exclude '.env' \
  --exclude '.env.local' \
  --exclude '.env.production' \
  --exclude '.env.development' \
  --exclude '/.venv*/' \
  --exclude '/sightindex.db' \
  --exclude '/*.db' \
  --exclude '/*.db-wal' \
  --exclude '/*.db-shm' \
  --exclude '/*.sqlite' \
  --exclude '/*.sqlite3' \
  --exclude '*.db' \
  --exclude '*.db-wal' \
  --exclude '*.db-shm' \
  --exclude '*.sqlite' \
  --exclude '*.sqlite3' \
  --exclude '*.sqlite-wal' \
  --exclude '*.sqlite-shm' \
  --exclude '/data/' \
  --exclude '/models/' \
  --exclude '/*.pt' \
  --exclude '/*.pth' \
  --exclude '/*.onnx' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.onnx' \
  --exclude '*.safetensors' \
  --exclude '*.ckpt' \
  --exclude '/docs/overall-effect-assessment-20260908.md' \
  --exclude '/docs/overall-effect-assessment-20260908.docx' \
  --exclude '/SOURCE_MANIFEST.json' \
  --exclude '/reid-calibration-report*.json' \
  --exclude '/reid-feedback*.csv' \
  --exclude '/logs/' \
  --exclude '/backups/' \
  --exclude '/frontend/node_modules/' \
  --exclude '/frontend/tsconfig.tsbuildinfo' \
  --exclude '/test-results/' \
  --exclude '/.pytest_cache/' \
  --exclude '/.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.sightindex-model-assets.lock' \
  --exclude '.sightindex-model-assets-quarantine/' \
  --exclude '*.partial' \
  "$ROOT_DIR/" "$target:$remote_dir/"

printf 'code synced to %s:%s; runtime data and databases were preserved\n' "$target" "$remote_dir"
