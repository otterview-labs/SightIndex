from __future__ import annotations

import os
import re
import subprocess
from ipaddress import ip_address
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RTX_DIR = ROOT / "deploy" / "rtx5090"


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


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
    assert 'APP_HOST=127.0.0.1' in template
    assert 'APP_PORT=8000' in template
    assert 'REID_SERVICE_HOST=127.0.0.1' in template
    assert 'MILVUS_BIND=127.0.0.1' in template
    assert 'MINIO_ROOT_PASSWORD=replace-with-random-secret' in template
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
    assert 'ENV_FILE must not be a symbolic link' in installer
    assert 'ENV_FILE must already be owned by root' in installer
    assert 'ENV_FILE must not be writable by $SERVICE_USER' in installer
    assert installer.index('test -w "$ROOT_DIR"') < installer.index(
        'chown "root:$SERVICE_GROUP" "$ENV_FILE"'
    )
    assert 'find "$ROOT_DIR" -xdev' in installer
    assert "-o -writable -print -quit" in installer
    assert 'status --porcelain --untracked-files=all' in installer
    assert 'DATA_DIR must resolve exactly to $SERVICE_HOME/data' in installer
    assert "must be a direct child of $SERVICE_HOME/db" in installer
    assert 'SERVICE_HOME must not be a symbolic link' in installer
    assert 'POSTGRES_BIND must be 127.0.0.1' in installer
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
    assert 'scripts/check_milvus.py --object-type reid_person_crop' in verifier
    assert 'Image.new("RGB", (640, 640)' in verifier
    assert 'run this verifier as $SERVICE_USER' in verifier
    assert 'HOME must be $SERVICE_HOME' in verifier
    assert 'bash "$ENV_FILE" "$python_bin" "$@"' in verifier
    assert "use KEY=value without export" in verifier
    assert "value must begin immediately after =" in verifier


def test_rtx_scripts_are_executable() -> None:
    for script in (RTX_DIR / "install_or_update.sh", RTX_DIR / "verify.sh"):
        assert os.access(script, os.X_OK)


def test_systemd_foreground_reid_does_not_need_repository_write_access() -> None:
    launcher = _read("deploy/agx/start_reid_service.sh")

    foreground = launcher.index('if [ "${REID_SERVICE_FOREGROUND:-0}" = "1" ]')
    background_log_directory = launcher.index("mkdir -p logs")
    assert foreground < background_log_directory
