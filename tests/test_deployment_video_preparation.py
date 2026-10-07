"""Video deployment checks use configuration templates, mocks and synthetic pixels only."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.config.settings import Settings
from deploy.containers import preflight, verify

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DEFAULTS = {
    "VIDEO_PREPARATION_ENABLED": "false",
    "VIDEO_FFMPEG_BINARY": "ffmpeg",
    "VIDEO_FFPROBE_BINARY": "ffprobe",
    "VIDEO_PREPARATION_TIMEOUT_SECONDS": "120",
    "VIDEO_PROBE_TIMEOUT_SECONDS": "10",
    "VIDEO_PREPARATION_MAX_DURATION_SECONDS": "300",
    "VIDEO_PREPARATION_MAX_PIXELS": "2073600",
    "VIDEO_PREPARATION_MAX_OUTPUT_BYTES": "134217728",
    "VIDEO_PREPARATION_MIN_FREE_BYTES": "536870912",
    "VIDEO_PREPARATION_THREADS": "2",
    "VIDEO_PREPARATION_INCLUDE_AUDIO": "false",
}


@pytest.mark.parametrize(
    "template",
    ["deploy/containers/.env.example", "deploy/rtx5090/sightindex.env.example"],
)
def test_video_templates_match_settings_opt_in_and_bounded_defaults(template: str) -> None:
    """Both targets expose the exact application fields, without enabling preparation."""
    values = {
        key: value
        for line in (ROOT / template).read_text().splitlines()
        if line and not line.startswith("#")
        for key, value in [line.split("=", 1)]
        if key in VIDEO_DEFAULTS
    }
    assert values == VIDEO_DEFAULTS
    settings = Settings(_env_file=None, **{key.lower(): value for key, value in values.items()})
    for key in VIDEO_DEFAULTS:
        name = key.lower()
        assert getattr(settings, name) == Settings.model_fields[name].default


def test_video_compose_passes_all_settings_only_to_api_and_image_has_tools() -> None:
    """The runtime image owns ffmpeg dependencies; no host mount or new service is added."""
    services = yaml.safe_load((ROOT / "deploy/containers/compose.yaml").read_text())["services"]
    environment = services["api"]["environment"]
    for key, default in VIDEO_DEFAULTS.items():
        assert environment[key] == "${" + key + ":-" + default + "}"
        assert key not in services["reid"]["environment"]
    dockerfile = (ROOT / "deploy/containers/Dockerfile").read_text()
    assert "libgl1 ffmpeg" in dockerfile
    assert "apt-get purge -y --auto-remove build-essential python3-dev" in dockerfile
    assert set(services) == {"postgres", "etcd", "minio", "milvus", "api", "reid"}


def _video_settings(enabled: bool, **overrides: Any) -> Settings:
    """Supply exact local fields without loading a developer's private env file."""
    return Settings(
        _env_file=None,
        video_preparation_enabled=enabled,
        video_ffmpeg_binary="ffmpeg",
        video_ffprobe_binary="ffprobe",
        app_basic_auth_username=None,
        app_basic_auth_password=None,
        **overrides,
    )


def test_disabled_runtime_video_preparation_does_not_inspect_or_run_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled installation never requires media tool availability."""
    monkeypatch.setattr(verify.shutil, "which", lambda _tool: pytest.fail("tool lookup"))
    monkeypatch.setattr(verify.subprocess, "run", lambda *a, **k: pytest.fail("tool command"))
    assert "disabled" in verify.verify_video_preparation_tools(_video_settings(False))


def test_enabled_runtime_video_tools_only_run_bounded_version_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Version commands cannot accidentally open or echo any business media."""
    calls: list[list[str]] = []
    monkeypatch.setattr(verify.shutil, "which", lambda tool: f"/synthetic/bin/{tool}")

    def run(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        assert kwargs == {
            "check": False,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "timeout": 10,
        }
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(verify.subprocess, "run", run)
    assert "no media decoded" in verify.verify_video_preparation_tools(_video_settings(True))
    assert calls == [
        ["/synthetic/bin/ffmpeg", "-version"],
        ["/synthetic/bin/ffprobe", "-version"],
    ]


@pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
def test_enabled_runtime_missing_video_tools_fail_without_echoing_configuration(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """A fixed label is safe even when the configured command contains private paths."""
    settings = _video_settings(True)
    settings.video_ffmpeg_binary = "/private-sensitive/ffmpeg"
    settings.video_ffprobe_binary = "/private-sensitive/ffprobe"
    monkeypatch.setattr(
        verify.shutil, "which", lambda tool: None if tool.endswith(missing) else "/synthetic/tool"
    )
    monkeypatch.setattr(
        verify.subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(args, 0)
    )
    with pytest.raises(verify.AcceptanceError, match=f"executable {missing}") as error:
        verify.verify_video_preparation_tools(settings)
    assert "private-sensitive" not in str(error.value)


@pytest.mark.parametrize("configured", ["", "secret\nffmpeg", "secret\rffmpeg", "secret\x00ffmpeg"])
def test_enabled_runtime_invalid_video_tool_names_do_not_execute(
    monkeypatch: pytest.MonkeyPatch, configured: str
) -> None:
    settings = _video_settings(True)
    settings.video_ffmpeg_binary = configured
    monkeypatch.setattr(verify.shutil, "which", lambda _tool: pytest.fail("tool lookup"))
    with pytest.raises(verify.AcceptanceError, match="executable ffmpeg") as error:
        verify.verify_video_preparation_tools(settings)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("failure", ["exit", "timeout", "oserror"])
def test_enabled_runtime_video_tool_failure_is_bounded_and_sanitized(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr(verify.shutil, "which", lambda _tool: "/private-sensitive/tool")

    def run(arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(arguments, 10, output="sensitive decoder text")
        if failure == "oserror":
            raise OSError("sensitive decoder text")
        return subprocess.CompletedProcess(arguments, 1, "sensitive decoder text", "private path")

    monkeypatch.setattr(verify.subprocess, "run", run)
    with pytest.raises(verify.AcceptanceError, match="tool check failed: ffmpeg") as error:
        verify.verify_video_preparation_tools(_video_settings(True))
    assert "sensitive" not in str(error.value)
    assert "private" not in str(error.value)


@pytest.mark.parametrize("enabled,tools_available", [(False, False), (True, True), (True, False)])
def test_runtime_cli_runs_video_tool_gate_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    enabled: bool,
    tools_available: bool,
) -> None:
    """Tool failures prevent acceptance only for the explicitly opted-in capability."""
    import app.config.settings as settings_module

    settings = _video_settings(enabled)
    monkeypatch.setattr(settings_module, "Settings", lambda: settings)
    monkeypatch.setattr(
        verify, "verify_runtime", lambda *_args: [verify.CheckResult("fixture", True, "safe")]
    )
    calls: list[bool] = []

    def tools(_settings: Settings) -> str:
        calls.append(True)
        if not tools_available:
            raise verify.AcceptanceError("video preparation requires executable ffmpeg")
        return "mocked tools"

    monkeypatch.setattr(verify, "verify_video_preparation_tools", tools)
    assert verify.main(["--stacks", "base"]) == (1 if enabled and not tools_available else 0)
    assert bool(calls) == enabled
    output = capsys.readouterr()
    assert ("video preparation tools" in output.out) == enabled


def test_isolated_upload_smoke_disables_ambient_video_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The baseline generated MJPG upload does not opt in via inherited deployment env."""
    import app.services.video_processing as processing

    monkeypatch.setenv("VIDEO_PREPARATION_ENABLED", "true")
    monkeypatch.setattr(
        processing, "prepare_uploaded_video", lambda *_args: pytest.fail("unexpected preparation")
    )
    assert "five frames/crops" in verify.isolated_upload_smoke()


@pytest.mark.parametrize(
    "target,enabled", [("containers", False), ("containers", True), ("rtx5090", False)]
)
def test_host_preflight_does_not_require_video_tools_when_container_owned_or_disabled(
    monkeypatch: pytest.MonkeyPatch, target: str, enabled: bool
) -> None:
    """A container carries its tools; a disabled native deployment does not need them."""
    checked: list[str] = []

    def which(tool: str) -> str | None:
        checked.append(tool)
        return None if tool in {"ffmpeg", "ffprobe"} else f"/synthetic/{tool}"

    monkeypatch.setattr(preflight.shutil, "which", which)
    monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
    monkeypatch.setattr(preflight.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, "v22.18.0" if args[0] == "node" else "linux", ""
        ),
    )
    errors = preflight.check_tools(
        target, {"base"}, {"VIDEO_PREPARATION_ENABLED": "true" if enabled else "false"}
    )
    assert errors == []
    assert not {"ffmpeg", "ffprobe"} & set(checked)


@pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
def test_native_preflight_requires_configured_tools_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """Native preflight gates missing executable tools before any deployment writes."""
    checked: list[str] = []

    def which(tool: str) -> str | None:
        checked.append(tool)
        return None if tool.endswith(missing) else f"/synthetic/{tool}"

    monkeypatch.setattr(preflight.shutil, "which", which)
    monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
    monkeypatch.setattr(preflight.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, "v22.18.0" if args[0] == "node" else "linux", ""
        ),
    )
    errors = preflight.check_tools(
        "rtx5090",
        {"base"},
        {
            "VIDEO_PREPARATION_ENABLED": "true",
            "VIDEO_FFMPEG_BINARY": "/private-sensitive/ffmpeg",
            "VIDEO_FFPROBE_BINARY": "/private-sensitive/ffprobe",
        },
    )
    assert any(missing in error for error in errors)
    assert "/private-sensitive/ffmpeg" in checked
    assert "/private-sensitive/ffprobe" in checked
    assert "private-sensitive" not in " ".join(errors)
