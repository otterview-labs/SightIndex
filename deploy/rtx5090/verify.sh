#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
SERVICE_USER="sightindex"
SERVICE_HOME="/var/lib/sightindex"
cd "$ROOT_DIR"

log() {
  printf '[sightindex-verify] %s\n' "$*"
}

fail() {
  printf '[sightindex-verify] ERROR: %s\n' "$*" >&2
  exit 1
}

[ -f "$ENV_FILE" ] || fail "$ENV_FILE is missing"
[ -x "$ROOT_DIR/.venv/bin/python" ] || fail "$ROOT_DIR/.venv/bin/python is missing"
[ "$(id -un)" = "$SERVICE_USER" ] \
  || fail "run this verifier as $SERVICE_USER"
[ "${HOME:-}" = "$SERVICE_HOME" ] \
  || fail "HOME must be $SERVICE_HOME"
python_bin="$ROOT_DIR/.venv/bin/python"

parsed_env="$(mktemp)"
if ! python3 - "$ENV_FILE" >"$parsed_env" <<'PY'
import re
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1])
seen: set[str] = set()
for line_number, original in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    stripped = original.strip()
    if not stripped or stripped.startswith("#"):
        continue
    if stripped.startswith("export "):
        raise SystemExit(f"{path}:{line_number}: use KEY=value without export")
    key, separator, raw_value = stripped.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise SystemExit(f"{path}:{line_number}: expected KEY=value without whitespace")
    if raw_value[:1].isspace():
        raise SystemExit(f"{path}:{line_number}: value must begin immediately after =")
    if key in {
        "BASHOPTS", "BASH_ENV", "CDPATH", "ENV", "GLOBIGNORE", "HOME", "IFS",
        "LOGNAME", "PATH", "SHELL", "SHELLOPTS", "USER",
    }:
        raise SystemExit(f"{path}:{line_number}: reserved process variable {key}")
    if key in seen:
        raise SystemExit(f"{path}:{line_number}: duplicate key {key}")
    if any(character in raw_value for character in "$`\\;|&<>()#~"):
        raise SystemExit(
            f"{path}:{line_number}: shell expansion and control characters are not supported"
        )
    lexer = shlex.shlex(raw_value, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    parts = list(lexer)
    if len(parts) > 1:
        raise SystemExit(f"{path}:{line_number}: quote values containing spaces")
    value = parts[0] if parts else ""
    if any(character in value for character in "\t\r\n"):
        raise SystemExit(f"{path}:{line_number}: tabs and newlines are not supported")
    seen.add(key)
    print(f"{key}\t{value}")
PY
then
  rm -f "$parsed_env"
  fail "$ENV_FILE is not a valid deployment environment file"
fi
while IFS=$'\t' read -r key value; do
  printf -v "CONFIG_$key" '%s' "$value"
done <"$parsed_env"
rm -f "$parsed_env"

cfg() {
  local key="$1"
  local default_value="${2-}"
  local variable_name="CONFIG_$key"
  if [ "${!variable_name+x}" = x ]; then
    printf '%s' "${!variable_name}"
  else
    printf '%s' "$default_value"
  fi
}

has_cfg() {
  local variable_name="CONFIG_$1"
  [ "${!variable_name+x}" = x ]
}

is_true() {
  case "$1" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

run_python() {
  env -i \
    "HOME=$SERVICE_HOME" \
    "USER=$SERVICE_USER" \
    "LOGNAME=$SERVICE_USER" \
    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    "LANG=C" \
    bash -c '
      env_file="$1"
      python_bin="$2"
      shift 2
      set -a
      . "$env_file"
      set +a
      exec "$python_bin" "$@"
    ' bash "$ENV_FILE" "$python_bin" "$@"
}

app_port="$(cfg APP_PORT 8000)"
reid_port="$(cfg REID_SERVICE_PORT 18031)"
data_dir="$(cfg DATA_DIR /var/lib/sightindex/data)"
if [[ "$data_dir" != /* ]]; then
  data_dir="$ROOT_DIR/${data_dir#./}"
fi
has_cfg APP_HOST && [ "$(cfg APP_HOST)" = "127.0.0.1" ] \
  || fail "APP_HOST must be explicitly set to 127.0.0.1"
has_cfg APP_PORT && [[ "$app_port" =~ ^[0-9]+$ ]] \
  || fail "APP_PORT must be explicitly set to an integer"

temporary_dir="$(mktemp -d)"
trap 'rm -rf "$temporary_dir"' EXIT
curl_auth=()
if [ -n "$(cfg APP_BASIC_AUTH_USERNAME)" ] || [ -n "$(cfg APP_BASIC_AUTH_PASSWORD)" ]; then
  [ -n "$(cfg APP_BASIC_AUTH_USERNAME)" ] && [ -n "$(cfg APP_BASIC_AUTH_PASSWORD)" ] \
    || fail "both APP_BASIC_AUTH_USERNAME and APP_BASIC_AUTH_PASSWORD must be set"
  username="$(cfg APP_BASIC_AUTH_USERNAME)"
  password="$(cfg APP_BASIC_AUTH_PASSWORD)"
  [[ "$username" != *:* && "$username" != *\\* && "$username" != *\"* ]] \
    || fail "APP_BASIC_AUTH_USERNAME contains a character unsupported by the verifier"
  [[ "$password" != *\\* && "$password" != *\"* ]] \
    || fail "APP_BASIC_AUTH_PASSWORD cannot contain a backslash or double quote"
  curl_config="$temporary_dir/curl-auth.conf"
  printf 'user = "%s:%s"\n' "$username" "$password" >"$curl_config"
  chmod 0600 "$curl_config"
  curl_auth=(--config "$curl_config")
fi

health_file="$temporary_dir/health.json"
reid_ready_file="$temporary_dir/reid-ready.json"
reid_status_file="$temporary_dir/reid-status.json"
media_counts_file="$temporary_dir/media-counts.json"
openapi_file="$temporary_dir/openapi.json"
face_probe_file="$temporary_dir/face-probe.json"

command -v systemctl >/dev/null 2>&1 || fail "systemd is required for this deployment profile"
services=(sightindex-api.service)
if is_true "$(cfg REID_ENABLED false)"; then
  services=(sightindex-reid.service "${services[@]}")
fi
for service in "${services[@]}"; do
  systemctl is-active --quiet "$service" || fail "$service is not active"
  restarts="$(systemctl show "$service" -p NRestarts --value)"
  log "$service active (restart count: ${restarts:-unknown})"
done

log "checking CUDA execution as $(id -un)"
run_python - <<'PY'
import onnxruntime as ort
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA")
device = torch.cuda.get_device_name(0)
if "5090" not in device:
    raise SystemExit(f"expected RTX 5090, got {device!r}")
value = (torch.ones(1024, device="cuda") * 2).sum().item()
torch.cuda.synchronize()
if value != 2048:
    raise SystemExit("unexpected CUDA tensor result")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("onnxruntime-gpu does not expose CUDAExecutionProvider")
print(f"CUDA runtime: PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {device}")
PY

if is_true "$(cfg REID_ENABLED false)"; then
  curl --noproxy '*' -fsS --max-time 5 \
    "http://127.0.0.1:$reid_port/ready" >"$reid_ready_file" \
    || fail "ReID /ready failed"
  run_python - "$reid_ready_file" \
    "$(cfg REID_MODEL sapiensid_wb12m)" \
    "$(cfg REID_CHECKPOINT_REVISION)" \
    "$(cfg REID_EMBEDDING_DIM 4096)" \
    "$(cfg REID_PREPROCESS_VERSION squarepad-v1)" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
ready = payload.get("ready")
if ready is None:
    ready = (
        payload.get("status") == "ok"
        and payload.get("loaded") is True
        and not payload.get("warmup_error")
        and payload.get("checkpoint_present") is True
        and payload.get("config_present") is True
        and payload.get("pipeline_assets_present") is True
        and not payload.get("missing_assets")
    )
if not ready:
    raise SystemExit("ReID runtime is not ready")
if not str(payload.get("device", "")).startswith("cuda"):
    raise SystemExit("ReID runtime is not using CUDA")
required = ("model", "checkpoint_revision", "embedding_dim", "preprocess_version")
missing = [key for key in required if not payload.get(key)]
if missing:
    raise SystemExit(f"ReID identity is incomplete: {', '.join(missing)}")
expected = {
    "model": sys.argv[2],
    "checkpoint_revision": sys.argv[3],
    "embedding_dim": int(sys.argv[4]),
    "preprocess_version": sys.argv[5],
}
mismatched = [key for key, value in expected.items() if payload.get(key) != value]
if mismatched:
    raise SystemExit(f"ReID identity does not match .env: {', '.join(mismatched)}")
print(
    "ReID model identity:",
    payload["model"],
    payload["embedding_dim"],
    payload["preprocess_version"],
)
PY
fi

curl --noproxy '*' -fsS --max-time 5 \
  "http://127.0.0.1:$app_port/health" >"$health_file" \
  || fail "API /health failed"
run_python - "$health_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
if payload != {"status": "ok"}:
    raise SystemExit("API /health returned an unexpected payload")
PY
curl --noproxy '*' -fsS --max-time 10 "${curl_auth[@]}" \
  "http://127.0.0.1:$app_port/api/media/counts" >"$media_counts_file" \
  || fail "database-backed media count check failed"
run_python - "$media_counts_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
required = ("image_with_crops_count", "person_crop_count")
for key in required:
    if not isinstance(payload.get(key), int) or payload[key] < 0:
        raise SystemExit(f"invalid media count field: {key}")
print(
    f"Database-backed media counts: {payload['image_with_crops_count']} images, "
    f"{payload['person_crop_count']} person crops"
)
PY

if is_true "$(cfg REID_ENABLED false)"; then
  curl --noproxy '*' -fsS --max-time 10 "${curl_auth[@]}" \
    "http://127.0.0.1:$app_port/api/reid/status" >"$reid_status_file" \
    || fail "API ReID status failed"
  require_face_priority=0
  if is_true "$(cfg REID_FACE_PRIORITY_ENABLED false)"; then
    require_face_priority=1
  fi
  run_python - "$reid_status_file" "$require_face_priority" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
checks = {
    "enabled": payload.get("enabled"),
    "ready": payload.get("ready"),
    "reid_service_ok": payload.get("reid_service_ok"),
    "milvus_configured": payload.get("milvus_configured"),
    "milvus_ok": payload.get("milvus_ok"),
}
if sys.argv[2] == "1":
    checks["face_priority_ready"] = payload.get("face_priority_ready")
failed = [name for name, passed in checks.items() if not passed]
if failed:
    detail = payload.get("last_error") or payload.get("face_priority_error") or "no detail"
    raise SystemExit(f"ReID capability check failed ({', '.join(failed)}): {detail}")
rescue_floor = float(payload.get("face_rescue_min_body_score", -1))
body_floor = min(
    float(payload.get("min_score", 1)),
    float(payload.get("min_score_cross_camera", 1)),
)
if not 0 <= rescue_floor <= body_floor:
    raise SystemExit(
        f"invalid face rescue floor {rescue_floor}; expected 0 <= rescue <= {body_floor}"
    )
print(
    f"ReID coverage: {payload.get('indexed_crops', 0)} indexed, "
    f"{payload.get('pending_crops', 0)} pending; "
    f"face={payload.get('face_model') or 'disabled'} on {payload.get('face_device') or 'n/a'}; "
    f"face rescue floor={rescue_floor:.2f}"
)
PY
fi

curl --noproxy '*' -fsS --max-time 10 "${curl_auth[@]}" \
  "http://127.0.0.1:$app_port/openapi.json" >"$openapi_file" \
  || fail "OpenAPI contract is unavailable"
run_python - "$openapi_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    paths = json.load(stream).get("paths", {})
required = {
    "/api/media/counts",
    "/api/reid/status",
    "/api/reid/search",
    "/api/reid/crops/{crop_id}/similar",
    "/api/reid/crops/{crop_id}/links",
    "/api/reid/feedback",
    "/api/reid/feedback/export.csv",
    "/api/reid/index/rebuild",
    "/api/face/library/rebuild",
    "/api/attributes/person-crops/backfill",
    "/api/attributes/jobs",
    "/api/attributes/jobs/{crop_id}/retry",
    "/api/search/observations/rebuild",
}
missing = sorted(required - paths.keys())
if missing:
    raise SystemExit(f"deployed API is missing routes: {', '.join(missing)}")
feedback_methods = set(paths["/api/reid/feedback"])
if not {"get", "put"}.issubset(feedback_methods):
    raise SystemExit(
        "deployed ReID feedback route must expose both GET and PUT; found "
        + ", ".join(sorted(feedback_methods))
    )
print(f"Critical API routes: {len(required)}/{len(required)} present")
schemas = contract.get("components", {}).get("schemas", {})
for response in ("ReidSearchResponse", "ReidLinkResponse"):
    if "face_coverage" not in schemas.get(response, {}).get("properties", {}):
        raise SystemExit(f"deployed {response} is missing per-query face diagnostics")
required_coverage = {"status", "shortlist_count", "compared_count", "query_absence_reasons",
                     "candidate_absence_reasons", "query_identity_verified"}
if not required_coverage <= schemas.get("ReidFaceCoverage", {}).get("properties", {}).keys():
    raise SystemExit("deployed face diagnostic contract is incomplete")
print("Per-query face diagnostic contracts: present (not proof of face coverage/accuracy)")
PY

# Read-only schema check: a healthy process must not hide a skipped additive migration.
.venv/bin/python - <<'PY'
from sqlalchemy import inspect
from app.db.session import engine

columns = {column["name"] for column in inspect(engine).get_columns("crop_face_extractions")}
if not {"absence_reason", "input_fingerprint"} <= columns:
    raise SystemExit("face cache schema is outdated; back up the DB and run init_db() before use")
print("Face cache schema: reason and input fingerprint columns present")
PY

if is_true "$(cfg MILVUS_ENABLED false)"; then
  curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:9091/healthz >/dev/null \
    || fail "Milvus health endpoint failed"
  run_python scripts/check_milvus.py --object-type reid_person_crop
fi

if is_true "$(cfg REID_FACE_PRIORITY_ENABLED false)"; then
  synthetic_image="$temporary_dir/face-runtime-probe.png"
  run_python - "$synthetic_image" <<'PY'
import sys
from PIL import Image

# Synthetic pixels initialize the complete InsightFace ONNX session without reading biometric
# data from the deployment. Detecting zero faces is expected; model/session load is the check.
Image.new("RGB", (640, 640), color=(96, 96, 96)).save(sys.argv[1])
PY
  log "initializing the InsightFace CUDA session with a synthetic image"
  run_python scripts/face_cuda_probe.py "$synthetic_image" >"$face_probe_file"
fi

curl --noproxy '*' -fsS --max-time 10 "${curl_auth[@]}" \
  "http://127.0.0.1:$app_port/reid" \
  | grep -qi '<!doctype html>' \
  || fail "frontend /reid page is not being served"

# Check the real read heartbeat without printing RTSP URLs or credentials.
.venv/bin/python - "$app_port" <<'PY'
import base64
import json
import os
import sys
from urllib.request import Request, urlopen

def read(path):
    request = Request(f"http://127.0.0.1:{sys.argv[1]}{path}")
    user = os.environ.get("APP_BASIC_AUTH_USERNAME")
    password = os.environ.get("APP_BASIC_AUTH_PASSWORD")
    if user and password:
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        request.add_header("Authorization", f"Basic {token}")
    with urlopen(request, timeout=10) as response:
        return json.load(response)

streams = read("/api/streams")
active = [stream for stream in streams if stream["status"] != "stopped"]
for stream in streams:
    print("Capture:", stream["name"], stream.get("capture_health"),
          "last read:", stream.get("last_frame_read_at"),
          "read failures:", stream.get("consecutive_read_failures"))
if any(stream.get("capture_health") != "healthy" for stream in active):
    raise SystemExit("An enabled camera has no fresh successful-read heartbeat")
if not active:
    print("No enabled cameras: capture capability requires a live camera check")
jobs = read("/api/attributes/jobs")
print("Live VLM coverage:", jobs["pending_crops"], "pending crops; queue:", jobs["queue"])
PY

if [ -f "$data_dir/tasks/attribute-backfill.json" ]; then
  log "attribute backfill checkpoint exists under DATA_DIR/tasks"
fi

revision="$(git -c safe.directory="$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || printf unknown)"
log "all deployment checks passed at revision $revision"
log "live-camera frame progression and external TLS access still require an operator check"
