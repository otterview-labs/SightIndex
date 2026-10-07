from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from ipaddress import ip_address
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RTX_DIR = ROOT / "deploy" / "rtx5090"


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _verifier_heredoc(command: str, *, contains: str | None = None) -> str:
    matches = re.findall(
        re.escape(command) + r" <<'PY'\n(.*?)\nPY\n",
        _read("deploy/rtx5090/verify.sh"),
        re.DOTALL,
    )
    if contains is not None:
        matches = [body for body in matches if contains in body]
    assert len(matches) == 1, f"expected a single verifier heredoc for {command}"
    return matches[0]


def _synthetic_openapi() -> dict:
    from app.schemas.media import VideoPlaybackRead

    routes = (
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
        "/api/images/{image_id}/playback",
        "/api/person-crops/{crop_id}/playback",
    )
    paths = {route: {"get": {}} for route in routes}
    paths["/api/reid/feedback"]["put"] = {}
    for route in ("/api/images/{image_id}/playback", "/api/person-crops/{crop_id}/playback"):
        paths[route]["get"] = {
            "responses": {
                "200": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/VideoPlaybackRead",
                            }
                        }
                    }
                }
            },
        }
    return {
        "paths": paths,
        "components": {
            "schemas": {
                "VideoPlaybackRead": VideoPlaybackRead.model_json_schema(),
                "ReidSearchResponse": {"properties": {"face_coverage": {}}},
                "ReidLinkResponse": {"properties": {"face_coverage": {}}},
                "ReidFaceCoverage": {
                    "properties": {
                        name: {}
                        for name in (
                            "status",
                            "shortlist_count",
                            "compared_count",
                            "query_absence_reasons",
                            "candidate_absence_reasons",
                            "query_identity_verified",
                        )
                    }
                },
            }
        },
    }


def _run_openapi_heredoc(tmp_path: Path, payload: dict) -> subprocess.CompletedProcess:
    fixture = tmp_path / "openapi.json"
    fixture.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-c", _verifier_heredoc('run_python - "$openapi_file"'), str(fixture)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_deployment_shell_scripts_parse() -> None:
    scripts = sorted((ROOT / "deploy").glob("**/*.sh"))
    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_rtx_profile_uses_public_deployment_boundaries() -> None:
    installer = _read("deploy/rtx5090/install_or_update.sh")
    verifier = _read("deploy/rtx5090/verify.sh")
    template = _read("deploy/rtx5090/sightindex.env.example")
    combined = "\n".join((installer, verifier, template))

    assert 'EXPECTED_ROOT="/opt/sightindex"' in installer
    assert 'SERVICE_USER="sightindex"' in installer
    assert 'DEPLOY_USER="sightindex-deploy"' in installer
    assert "APP_HOST=127.0.0.1" in template
    assert "APP_PORT=8000" in template
    assert "REID_SERVICE_HOST=127.0.0.1" in template
    assert "MILVUS_BIND=127.0.0.1" in template
    assert "MINIO_ROOT_PASSWORD=replace-with-random-secret" in template
    assert "replace the placeholder MINIO_ROOT_PASSWORD" in installer
    assert "--skip-deps requires an existing" in installer

    for forbidden in ("logs/uvicorn.log", "logs/reid_service.log"):
        assert forbidden not in combined

    assert not re.search(r"/home/[A-Za-z][A-Za-z0-9_-]*", combined)
    assert not re.search(r"\broot\s*@", combined)
    addresses = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", combined)
    for address in addresses:
        parsed = ip_address(address)
        assert not parsed.is_unspecified
        assert not parsed.is_private or parsed.is_loopback


def test_root_installer_does_not_execute_environment_file() -> None:
    installer = _read("deploy/rtx5090/install_or_update.sh")

    assert "source $ENV_FILE" not in installer
    assert not re.search(r"(?m)^\s*\.\s+[\"']?\$ENV_FILE", installer)
    assert 'chown "root:$SERVICE_GROUP" "$ENV_FILE"' in installer
    assert 'chmod 0640 "$ENV_FILE"' in installer
    assert 'if as_service test -w "$ROOT_DIR"' in installer
    assert "ENV_FILE must not be a symbolic link" in installer
    assert "ENV_FILE must already be owned by root" in installer
    assert "ENV_FILE must not be writable by $SERVICE_USER" in installer
    assert installer.index('test -w "$ROOT_DIR"') < installer.index(
        'chown "root:$SERVICE_GROUP" "$ENV_FILE"'
    )
    assert 'find "$ROOT_DIR" -xdev' in installer
    assert "-o -writable -print -quit" in installer
    assert "status --porcelain --untracked-files=all" in installer
    assert "DATA_DIR must resolve exactly to $SERVICE_HOME/data" in installer
    assert "must be a direct child of $SERVICE_HOME/db" in installer
    assert "SERVICE_HOME must not be a symbolic link" in installer
    assert "POSTGRES_BIND must be 127.0.0.1" in installer
    assert "DATABASE_URL does not target the running repository PostgreSQL" in installer
    assert "is still active; refusing to mutate the deployment" in installer
    assert "assert_trusted_artifact" in installer
    assert "contains a path writable by $SERVICE_USER" in installer
    assert "is-enabled --quiet sightindex-embedding.service" in installer
    assert 'runuser -u "$SERVICE_USER" -- env -i' in installer
    assert "as_service_configured" in installer
    assert 'bash "$ENV_FILE" "$@"' in installer
    assert "use KEY=value without export" in installer
    assert "value must begin immediately after =" in installer
    assert "APP_HOST must be explicitly set" in installer
    assert "APP_PORT must be explicitly set" in installer
    assert "deploy/systemd/*.service" not in installer


def test_rtx_verifier_handles_current_reid_readiness_contract() -> None:
    verifier = _read("deploy/rtx5090/verify.sh")

    assert 'ready = payload.get("ready")' in verifier
    assert "if ready is None:" in verifier
    assert 'payload.get("loaded") is True' in verifier
    assert 'payload.get("pipeline_assets_present") is True' in verifier
    assert '"milvus_configured": payload.get("milvus_configured")' in verifier
    assert "scripts/check_milvus.py --object-type reid_person_crop" in verifier
    assert 'Image.new("RGB", (640, 640)' in verifier
    assert "run this verifier as $SERVICE_USER" in verifier
    assert "HOME must be $SERVICE_HOME" in verifier
    assert 'bash "$ENV_FILE" "$python_bin" "$@"' in verifier
    assert "use KEY=value without export" in verifier
    assert "value must begin immediately after =" in verifier


def test_rtx_openapi_heredoc_executes_current_contract(tmp_path: Path) -> None:
    result = _run_openapi_heredoc(tmp_path, _synthetic_openapi())

    assert result.returncode == 0, result.stderr
    assert "Critical API routes: 15/15 present" in result.stdout
    assert "Stored-video playback routes and response contract: present" in result.stdout
    assert "Per-query face diagnostic contracts: present" in result.stdout


@pytest.mark.parametrize(
    "route",
    [
        "/api/images/{image_id}/playback",
        "/api/person-crops/{crop_id}/playback",
    ],
)
def test_rtx_openapi_heredoc_rejects_old_api_without_playback(
    tmp_path: Path,
    route: str,
) -> None:
    payload = _synthetic_openapi()
    del payload["paths"][route]

    result = _run_openapi_heredoc(tmp_path, payload)

    assert result.returncode != 0
    assert f"deployed API is missing routes: {route}" in result.stderr


@pytest.mark.parametrize(
    "field",
    [
        "available",
        "source_type",
        "video_url",
        "offset_seconds",
        "captured_at",
        "reason",
    ],
)
def test_rtx_openapi_heredoc_rejects_missing_playback_fields(
    tmp_path: Path,
    field: str,
) -> None:
    payload = _synthetic_openapi()
    del payload["components"]["schemas"]["VideoPlaybackRead"]["properties"][field]

    result = _run_openapi_heredoc(tmp_path, payload)

    assert result.returncode != 0
    assert f"deployed VideoPlaybackRead is missing fields: {field}" in result.stderr


@pytest.mark.parametrize(
    "route",
    [
        "/api/images/{image_id}/playback",
        "/api/person-crops/{crop_id}/playback",
    ],
)
@pytest.mark.parametrize("missing", ["get", "response_schema"])
def test_rtx_openapi_heredoc_rejects_incomplete_playback_get_response(
    tmp_path: Path,
    route: str,
    missing: str,
) -> None:
    payload = _synthetic_openapi()
    if missing == "get":
        payload["paths"][route] = {"post": payload["paths"][route]["get"]}
    else:
        payload["paths"][route]["get"]["responses"]["200"]["content"] = {}

    result = _run_openapi_heredoc(tmp_path, payload)

    assert result.returncode != 0
    assert f"no VideoPlaybackRead GET response: {route}" in result.stderr


def test_rtx_openapi_heredoc_rejects_unbounded_playback_offset(tmp_path: Path) -> None:
    payload = _synthetic_openapi()
    payload["components"]["schemas"]["VideoPlaybackRead"]["properties"]["offset_seconds"] = {
        "anyOf": [{"type": "number"}, {"type": "null"}],
        "default": None,
        "title": "Offset Seconds",
    }

    result = _run_openapi_heredoc(tmp_path, payload)

    assert result.returncode != 0
    assert "finite nonnegative contract" in result.stderr


def test_rtx_openapi_heredoc_rejects_missing_playback_schema(tmp_path: Path) -> None:
    payload = _synthetic_openapi()
    del payload["components"]["schemas"]["VideoPlaybackRead"]

    result = _run_openapi_heredoc(tmp_path, payload)

    assert result.returncode != 0
    assert "deployed VideoPlaybackRead is missing fields:" in result.stderr


@pytest.mark.parametrize("field", ["available", "source_type"])
def test_rtx_openapi_heredoc_rejects_optional_playback_availability_or_source(
    tmp_path: Path,
    field: str,
) -> None:
    payload = _synthetic_openapi()
    payload["components"]["schemas"]["VideoPlaybackRead"]["required"].remove(field)

    result = _run_openapi_heredoc(tmp_path, payload)

    assert result.returncode != 0
    assert "missing required availability/source fields" in result.stderr


@pytest.mark.parametrize(
    "schema_name,field",
    [
        ("ReidSearchResponse", "face_coverage"),
        ("ReidLinkResponse", "face_coverage"),
        ("ReidFaceCoverage", "query_identity_verified"),
    ],
)
def test_rtx_openapi_heredoc_preserves_face_diagnostic_checks(
    tmp_path: Path,
    schema_name: str,
    field: str,
) -> None:
    payload = _synthetic_openapi()
    del payload["components"]["schemas"][schema_name]["properties"][field]

    result = _run_openapi_heredoc(tmp_path, payload)

    assert result.returncode != 0
    assert "face diagnostic" in result.stderr


@pytest.mark.parametrize(
    "missing_column",
    [
        None,
        "source_video_url",
        "video_offset_seconds",
        "processed_at",
        "absence_reason",
        "input_fingerprint",
    ],
)
def test_rtx_database_heredoc_requires_playback_completion_and_face_cache_columns(
    missing_column: str | None,
) -> None:
    columns = {
        "images": ["source_video_url", "video_offset_seconds", "processed_at"],
        "crop_face_extractions": ["absence_reason", "input_fingerprint"],
    }
    if missing_column is not None:
        for table_columns in columns.values():
            if missing_column in table_columns:
                table_columns.remove(missing_column)
    # Execute only the read-only heredoc with synthetic inspector modules; never import the real
    # database session, apply a migration, or execute the deployment verifier itself.
    bootstrap = """
import json
import sys
import types
columns = json.loads(sys.argv[1])
class Inspector:
    def get_columns(self, table):
        return [{"name": name} for name in columns[table]]
sqlalchemy = types.ModuleType("sqlalchemy")
sqlalchemy.inspect = lambda engine: Inspector()
session = types.ModuleType("app.db.session")
session.engine = object()
sys.modules["sqlalchemy"] = sqlalchemy
sys.modules["app.db.session"] = session
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            bootstrap
            + _verifier_heredoc("run_python -", contains="from sqlalchemy import inspect"),
            json.dumps(columns),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if missing_column is None:
        assert result.returncode == 0, result.stderr
        assert "Image schema:" in result.stdout
        assert "Face cache schema:" in result.stdout
    else:
        assert result.returncode != 0
        assert "schema is outdated" in result.stderr


@pytest.mark.parametrize("script", ["install_or_update.sh", "sync_code.sh"])
@pytest.mark.parametrize(
    "failed_step", [None, "test:reid", "test:search", "test:observations", "test:playback"]
)
def test_rtx_frontend_gate_runs_all_suites_and_propagates_failures(
    tmp_path: Path,
    script: str,
    failed_step: str | None,
) -> None:
    source = _read(f"deploy/rtx5090/{script}")
    if script == "install_or_update.sh":
        match = re.search(
            r'if \[ "\$SKIP_FRONTEND" -eq 0 \]; then\n  log "building frontend"\n.*?\nfi\n',
            source,
            re.DOTALL,
        )
    else:
        match = re.search(
            r'if \[ "\$skip_frontend" = false \]; then\n.*?\nfi\n',
            source,
            re.DOTALL,
        )
    assert match is not None
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls = tmp_path / "npm-calls"
    npm = binaries / "npm"
    npm.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >>"$RTX_TEST_NPM_CALLS"\n'
        'for argument in "$@"; do\n'
        '  if [ "$argument" = "$RTX_TEST_NPM_FAILURE" ]; then exit 37; fi\n'
        "done\n",
        encoding="utf-8",
    )
    npm.chmod(0o700)
    shell = """set -Eeuo pipefail
ROOT_DIR="$1"
LOG_DIR="$2"
SKIP_FRONTEND=0
skip_frontend=false
log() { :; }
as_deployer() { "$@"; }
""" + match.group(0)
    result = subprocess.run(
        ["bash", "-c", shell, "rtx-frontend-fixture", str(tmp_path), str(logs)],
        env={
            **os.environ,
            "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
            "RTX_TEST_NPM_CALLS": str(calls),
            "RTX_TEST_NPM_FAILURE": failed_step or "no-failure",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    executed = calls.read_text(encoding="utf-8").splitlines()
    steps = [line.split()[-1] for line in executed]
    expected = ["ci", "build", "test:reid", "test:search", "test:observations", "test:playback"]

    if failed_step is None:
        assert result.returncode == 0, result.stderr
        assert steps == expected
    else:
        assert result.returncode == 37
        assert steps == expected[: expected.index(failed_step) + 1]


def test_rtx_sync_default_destination_matches_installer_root() -> None:
    installer = _read("deploy/rtx5090/install_or_update.sh")
    sync = _read("deploy/rtx5090/sync_code.sh")
    expected_root = re.search(r'^EXPECTED_ROOT="([^"]+)"$', installer, re.MULTILINE)
    default_destination = re.search(r'^remote_dir="\$\{2:-([^}]+)}"$', sync, re.MULTILINE)

    assert expected_root is not None and default_destination is not None
    assert default_destination.group(1) == expected_root.group(1) == "/opt/sightindex"


def test_rtx_sync_preserves_nested_runtime_state_and_personal_reports() -> None:
    sync = _read("deploy/rtx5090/sync_code.sh")
    for pattern in (
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        "*.db",
        "*.db-wal",
        "*.db-shm",
        "*.sqlite",
        "*.sqlite3",
        "*.sqlite-wal",
        "*.sqlite-shm",
        "*.pt",
        "*.pth",
        "*.onnx",
        "*.safetensors",
        "*.ckpt",
        "/docs/overall-effect-assessment-20260908.md",
        "/docs/overall-effect-assessment-20260908.docx",
        "/SOURCE_MANIFEST.json",
        "/reid-calibration-report*.json",
        "/reid-feedback*.csv",
    ):
        assert f"--exclude '{pattern}'" in sync
    assert "StrictHostKeyChecking=yes" in sync
    assert "StrictHostKeyChecking=no" not in sync
    assert "--delete" not in sync


def test_model_recovery_state_is_excluded_from_build_context_and_code_sync() -> None:
    """Local recovery evidence must stay out of builds and remote code transfers."""
    dockerignore = _read(".dockerignore")
    sync = _read("deploy/rtx5090/sync_code.sh")
    rules = dockerignore.splitlines()
    last_allow = max(index for index, rule in enumerate(rules) if rule.startswith("!"))
    for rule in (
        "**/.sightindex-model-assets-quarantine",
        "**/.sightindex-model-assets-quarantine/**",
        "**/.sightindex-model-assets.lock",
    ):
        assert rule in rules
        assert rules.index(rule) > last_allow
    for pattern in (
        ".sightindex-model-assets-quarantine/",
        ".sightindex-model-assets.lock",
        "*.partial",
    ):
        assert f"--exclude '{pattern}'" in sync


def test_rtx_acceptance_http_client_is_an_explicit_runtime_dependency() -> None:
    """A clean RTX install must run its mandatory isolated TestClient smoke."""
    requirements = _read("requirements.rtx5090.txt")
    assert "httpx>=0.27,<1" in requirements.splitlines()
    verifier = _read("deploy/rtx5090/verify.sh")
    assert "--upload-smoke" in verifier


@pytest.mark.parametrize("preflight_exit", [0, 29])
def test_rtx_installer_check_delegates_before_mutations_and_preserves_exit(
    tmp_path: Path,
    preflight_exit: int,
) -> None:
    # A synthetic checkout and helper exercise argument routing without running any real install
    # step or loading a deployment environment. Its nonstandard root would fail normal install.
    checkout = tmp_path / "synthetic-checkout"
    installer = checkout / "deploy" / "rtx5090" / "install_or_update.sh"
    installer.parent.mkdir(parents=True)
    installer.write_text(_read("deploy/rtx5090/install_or_update.sh"), encoding="utf-8")
    helper = checkout / "deploy" / "containers" / "preflight.py"
    helper.parent.mkdir()
    args_file = tmp_path / "preflight-args.json"
    helper.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        'Path(os.environ["RTX_TEST_PREFLIGHT_ARGS"]).write_text(json.dumps(sys.argv[1:]))\n'
        'raise SystemExit(int(os.environ["RTX_TEST_PREFLIGHT_EXIT"]))\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(installer), "--check"],
        env={
            **os.environ,
            "SIGHTINDEX_DEPLOY_PYTHON": sys.executable,
            "RTX_TEST_PREFLIGHT_ARGS": str(args_file),
            "RTX_TEST_PREFLIGHT_EXIT": str(preflight_exit),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == preflight_exit, result.stderr
    assert json.loads(args_file.read_text(encoding="utf-8")) == [
        "--target",
        "rtx5090",
        "--source",
        str(checkout),
        "--env-file",
        str(checkout / ".env"),
        "--stacks",
        "base reid",
        "--check-tools",
    ]
    assert not (checkout / ".env").exists()
    assert sorted(path.name for path in checkout.iterdir()) == ["deploy"]
    assert "run as root" not in result.stderr
    assert "expected" not in result.stderr


def test_rtx_scripts_are_executable() -> None:
    for script in (RTX_DIR / "install_or_update.sh", RTX_DIR / "verify.sh"):
        assert os.access(script, os.X_OK)


def test_systemd_foreground_reid_does_not_need_repository_write_access() -> None:
    launcher = _read("deploy/agx/start_reid_service.sh")

    foreground = launcher.index('if [ "${REID_SERVICE_FOREGROUND:-0}" = "1" ]')
    background_log_directory = launcher.index("mkdir -p logs")
    assert foreground < background_log_directory


def test_public_documentation_has_language_switches() -> None:
    readme_en = _read("README.md")
    readme_zh = _read("README.zh-CN.md")
    deployment_en = _read("docs/deployment.md")
    deployment_zh = _read("docs/deployment.zh-CN.md")

    assert "[简体中文](README.zh-CN.md)" in readme_en
    assert "[English](README.md)" in readme_zh
    assert "[简体中文](deployment.zh-CN.md)" in deployment_en
    assert "[English](deployment.md)" in deployment_zh
