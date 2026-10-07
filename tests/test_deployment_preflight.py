"""Read-only preflight coverage uses private temporary config and synthetic model files only."""

from __future__ import annotations

import importlib.util
import json
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from urllib import request

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "deploy" / "containers" / "preflight.py"
SECRETS = (
    "synthetic-basic-secret",
    "synthetic-reid-key",
    "synthetic-postgres-secret",
    "synthetic-minio-secret",
    "synthetic-qwen-key",
    "synthetic-vlm-key",
)


@pytest.fixture
def preflight():
    spec = importlib.util.spec_from_file_location("sightindex_test_preflight", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _asset(path: Path, payload: bytes = b"synthetic-not-a-real-model") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_env(path: Path, values: dict[str, str]) -> None:
    path.write_text(
        "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _snapshot(directory: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(directory)): path.read_bytes() if path.is_file() else None
        for path in directory.rglob("*")
    }


@pytest.fixture
def deployment(tmp_path: Path):
    source = tmp_path / "source"
    _asset(source / "main.py")
    _asset(source / "deploy/containers/verify.py")
    models = tmp_path / "models"
    qwen = models / "qwen"
    assets = {
        "yolo": models / "yolo11n.pt",
        "reid_weights": models / "sapiensid_wb12m/model.pth",
        "reid_config": models / "sapiensid_wb12m/model.yaml",
        "pose": models / "yolov8n-pose.pt",
        "dfa": models / "dfa_mobilenetv4_medium/mobilenetv4_Final.pth",
        "face_detector": models / "insightface/models/buffalo_l/det_10g.onnx",
        "face_recognizer": models / "insightface/models/buffalo_l/w600k_r50.onnx",
        "qwen_config": qwen / "config.json",
        "qwen_tokenizer_config": qwen / "tokenizer_config.json",
        "qwen_preprocessor": qwen / "preprocessor_config.json",
        "qwen_tokenizer": qwen / "tokenizer.json",
        "qwen_weights": qwen / "model.safetensors",
        "qwen_script": qwen / "scripts/qwen3_vl_embedding.py",
    }
    for path in assets.values():
        _asset(path)
    values = {
        "SIGHTINDEX_IMAGE": "sightindex:synthetic-reviewed",
        "APP_BASIC_AUTH_USERNAME": "synthetic-operator",
        "APP_BASIC_AUTH_PASSWORD": SECRETS[0],
        "REID_SERVICE_API_KEY": SECRETS[1],
        "POSTGRES_PASSWORD": SECRETS[2],
        "MINIO_ROOT_PASSWORD": SECRETS[3],
        "QWEN_EMBEDDING_API_KEY": SECRETS[4],
        "VLM_API_KEY": SECRETS[5],
        "MODEL_DIR": str(models),
        "MEDIA_DIR": str(tmp_path / "media-not-yet-created"),
        "CACHE_DIR": str(tmp_path / "cache-not-yet-created"),
        "REID_ENABLED": "false",
        "SEMANTIC_SEARCH_ENABLED": "false",
        "REID_CHECKPOINT_REVISION": "sha256:" + "a" * 64,
        "QWEN_EMBEDDING_IMAGE": "sightindex-embedding:synthetic-reviewed",
        "QWEN_EMBEDDING_MODEL_DIR": str(qwen),
        "FACE_RECOGNITION_ON_INGEST": "false",
        "REID_FACE_PRIORITY_ENABLED": "true",
        "FACE_INSIGHTFACE_ALLOW_DOWNLOAD": "false",
        "VLM_PROVIDER": "none",
        "VLM_STRUCTURED_ON_INGEST": "false",
        "VLM_CAPTION_ON_INDEX": "false",
    }
    env_file = tmp_path / "private.env"
    _write_env(env_file, values)
    return source, values, env_file, assets


def _select(values: dict[str, str], stacks: set[str]) -> dict[str, str]:
    return {
        **values,
        "REID_ENABLED": "true" if "reid" in stacks else "false",
        "SEMANTIC_SEARCH_ENABLED": "true" if "semantic" in stacks else "false",
    }


def _cli(source: Path, env_file: Path, stacks: set[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--source",
            str(source),
            "--env-file",
            str(env_file),
            "--stacks",
            " ".join(sorted(stacks)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "stacks",
    [
        {"base"},
        {"base", "reid"},
        {"base", "embedding"},
        {"base", "embedding", "semantic"},
        {"base", "reid", "embedding", "semantic"},
    ],
)
def test_preflight_cli_accepts_complete_selected_stacks_without_mutation(
    deployment,
    stacks: set[str],
) -> None:
    source, values, env_file, _ = deployment
    _write_env(env_file, _select(values, stacks))
    before = _snapshot(env_file.parent)

    result = _cli(source, env_file, stacks)

    assert result.returncode == 0, result.stderr
    assert "Read-only preflight passed" in result.stdout
    assert ("body ReID" in result.stdout) == ("reid" in stacks)
    assert ("Qwen visual embedding" in result.stdout) == ("embedding" in stacks)
    assert ("semantic retrieval" in result.stdout) == ("semantic" in stacks)
    assert "VLM disabled" in result.stdout
    assert _snapshot(env_file.parent) == before
    assert all(secret not in result.stdout + result.stderr for secret in SECRETS)


def test_rtx_configuration_checks_checkout_gateway_and_assets_without_install(
    preflight,
    deployment,
    monkeypatch,
) -> None:
    source, values, _, assets = deployment
    _asset(source / preflight.DFA_RELATIVE)
    values = _select(values, {"base", "reid"})
    values.update(
        {
            "MILVUS_ENABLED": "true",
            "APP_HOST": "127.0.0.1",
            "PUBLIC_BASE_URL": "https://synthetic-gateway.test",
            "YOLO_MODEL": str(assets["yolo"]),
            "FACE_INSIGHTFACE_ROOT": str(assets["face_detector"].parents[2]),
            "REID_CHECKPOINT_DIR": str(assets["reid_weights"].parent),
            "REID_POSE_WEIGHTS": str(assets["pose"]),
            "VLM_STRUCTURED_BACKGROUND": "true",
        }
    )
    original_resolve = Path.resolve
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda path, *args, **kwargs: (
            Path("/opt/sightindex") if path == source else original_resolve(path, *args, **kwargs)
        ),
    )
    before = _snapshot(source.parent)

    errors, capabilities = preflight.check_configuration(
        "rtx5090", source, values, {"base", "reid"}
    )

    assert errors == []
    assert any("body ReID" in item for item in capabilities)
    assert _snapshot(source.parent) == before


@pytest.mark.parametrize(
    "asset_name",
    [
        "yolo",
        "reid_weights",
        "reid_config",
        "pose",
        "dfa",
        "face_detector",
        "face_recognizer",
        "qwen_config",
        "qwen_tokenizer_config",
        "qwen_preprocessor",
        "qwen_tokenizer",
        "qwen_weights",
        "qwen_script",
    ],
)
def test_preflight_rejects_missing_selected_model_assets(preflight, deployment, asset_name) -> None:
    source, values, _, assets = deployment
    assets[asset_name].unlink()
    stacks = {"base", "reid", "embedding", "semantic"}

    errors, _ = preflight.check_configuration("containers", source, _select(values, stacks), stacks)

    assert errors
    assert any("missing" in error for error in errors)
    assert all(secret not in " ".join(errors) for secret in SECRETS)


@pytest.mark.parametrize("asset_name", ["yolo", "qwen_weights", "qwen_tokenizer", "qwen_script"])
@pytest.mark.parametrize("invalid_kind", ["empty", "directory"])
def test_preflight_model_assets_must_be_nonempty_files(
    preflight,
    deployment,
    asset_name,
    invalid_kind,
) -> None:
    source, values, _, assets = deployment
    path = assets[asset_name]
    if invalid_kind == "empty":
        path.write_bytes(b"")
    else:
        path.unlink()
        path.mkdir()
    stacks = {"base", "embedding"}

    errors, _ = preflight.check_configuration("containers", source, _select(values, stacks), stacks)

    assert errors


@pytest.mark.parametrize(
    "calibration,expected",
    [
        ({}, True),
        (
            {
                "REID_CROSS_CAMERA_CALIBRATION_COEF": "1.5",
                "REID_CROSS_CAMERA_CALIBRATION_INTERCEPT": "-0.2",
            },
            True,
        ),
        ({"REID_CROSS_CAMERA_CALIBRATION_COEF": "1.5"}, False),
        ({"REID_CROSS_CAMERA_CALIBRATION_INTERCEPT": "0.2"}, False),
        (
            {
                "REID_CROSS_CAMERA_CALIBRATION_COEF": "",
                "REID_CROSS_CAMERA_CALIBRATION_INTERCEPT": "",
            },
            False,
        ),
        (
            {
                "REID_CROSS_CAMERA_CALIBRATION_COEF": "nan",
                "REID_CROSS_CAMERA_CALIBRATION_INTERCEPT": "0",
            },
            False,
        ),
        (
            {
                "REID_CROSS_CAMERA_CALIBRATION_COEF": "inf",
                "REID_CROSS_CAMERA_CALIBRATION_INTERCEPT": "0",
            },
            False,
        ),
        (
            {
                "REID_CROSS_CAMERA_CALIBRATION_COEF": "0",
                "REID_CROSS_CAMERA_CALIBRATION_INTERCEPT": "-inf",
            },
            False,
        ),
        (
            {
                "REID_CROSS_CAMERA_CALIBRATION_COEF": "1e309",
                "REID_CROSS_CAMERA_CALIBRATION_INTERCEPT": "0",
            },
            False,
        ),
        (
            {
                "REID_CROSS_CAMERA_CALIBRATION_COEF": "not-a-number",
                "REID_CROSS_CAMERA_CALIBRATION_INTERCEPT": "0",
            },
            False,
        ),
    ],
)
def test_preflight_calibration_is_an_optional_complete_finite_pair(
    preflight,
    deployment,
    calibration,
    expected,
) -> None:
    source, values, _, _ = deployment

    errors, _ = preflight.check_configuration(
        "containers", source, {**values, **calibration}, {"base"}
    )

    assert (errors == []) is expected
    if not expected:
        assert any("calibration" in error.lower() for error in errors)


@pytest.mark.parametrize(
    "line",
    [
        "SECRET=$(touch forbidden)",
        "SECRET=`touch forbidden`",
        "SECRET=x;touch",
        "SECRET=x|touch",
        "SECRET=x&touch",
        "SECRET=x<file",
        "SECRET=x>file",
        "SECRET=x\x00y",
        "SECRET=one\nSECRET=two",
        "PATH=untrusted",
        "HOME=untrusted",
        "CODEX_HOME=untrusted",
        "PYTHONPATH=untrusted",
        "LD_PRELOAD=untrusted",
        "export SECRET=untrusted",
        "SECRET='unterminated-synthetic-secret",
        "SECRET=unquoted spaces",
    ],
)
def test_preflight_env_rejects_unsafe_shell_duplicate_and_reserved_assignments(
    preflight,
    tmp_path,
    line,
) -> None:
    env_file = tmp_path / "private.env"
    env_file.write_text(line + "\n", encoding="utf-8")
    env_file.chmod(0o600)

    with pytest.raises(preflight.PreflightError):
        preflight.read_env(env_file)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["private.env"]


@pytest.mark.parametrize("invalid_kind", ["public_permissions", "symlink", "directory", "missing"])
def test_preflight_env_requires_a_private_regular_nonsymlink_file(
    preflight,
    tmp_path,
    invalid_kind,
) -> None:
    env_file = tmp_path / "private.env"
    if invalid_kind == "directory":
        env_file.mkdir()
    elif invalid_kind != "missing":
        _write_env(env_file, {"SECRET": "synthetic-secret"})
        if invalid_kind == "public_permissions":
            env_file.chmod(0o644)
        else:
            target = tmp_path / "linked.env"
            env_file.rename(target)
            env_file.symlink_to(target)

    with pytest.raises(preflight.PreflightError):
        preflight.read_env(env_file)


def test_preflight_env_supports_private_quotes_comments_and_blank_values(
    preflight, tmp_path
) -> None:
    env_file = tmp_path / "private.env"
    env_file.write_text(
        "# comment\nTOKEN='synthetic secret'\nEMPTY=\nMODE=true # comment\n", encoding="utf-8"
    )
    env_file.chmod(0o640)

    assert preflight.read_env(env_file) == {
        "TOKEN": "synthetic secret",
        "EMPTY": "",
        "MODE": "true",
    }


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {},
        {"weight_map": []},
        {"weight_map": "synthetic-secret"},
        {"weight_map": {"a": ["synthetic-secret"]}},
        {"weight_map": {"a": 123}},
        {"weight_map": {}},
    ],
)
def test_preflight_rejects_malformed_qwen_weight_manifest_without_traceback(
    deployment,
    manifest,
) -> None:
    source, values, env_file, assets = deployment
    _asset(
        assets["qwen_weights"].parent / "model.safetensors.index.json",
        json.dumps(manifest).encode(),
    )
    stacks = {"base", "embedding"}
    _write_env(env_file, _select(values, stacks))

    result = _cli(source, env_file, stacks)

    assert result.returncode == 1
    assert "Qwen weight manifest" in result.stderr
    assert "Traceback" not in result.stderr
    assert "synthetic-secret" not in result.stderr


@pytest.mark.parametrize(
    "shard",
    ["../outside.safetensors", "/outside.safetensors", "nested/weights.safetensors", ".", ".."],
)
def test_preflight_rejects_manifest_shard_path_traversal(preflight, deployment, shard) -> None:
    source, values, _, assets = deployment
    _asset(
        assets["qwen_weights"].parent / "model.safetensors.index.json",
        json.dumps({"weight_map": {"a": shard}}).encode(),
    )
    stacks = {"base", "embedding"}

    errors, _ = preflight.check_configuration("containers", source, _select(values, stacks), stacks)

    assert "Qwen manifest has an unsafe shard filename" in errors
    assert all(shard not in error for error in errors)


def test_preflight_shard_manifest_cannot_replace_runtime_monolithic_weights(
    preflight, deployment
) -> None:
    source, values, _, assets = deployment
    qwen = assets["qwen_weights"].parent
    shard = _asset(qwen / "model-00001-of-00001.safetensors")
    _asset(
        qwen / "model.safetensors.index.json",
        json.dumps({"weight_map": {"a": shard.name, "b": shard.name}}).encode(),
    )
    stacks = {"base", "embedding"}
    selected = _select(values, stacks)

    errors, _ = preflight.check_configuration("containers", source, selected, stacks)
    assert errors == []
    assets["qwen_weights"].unlink()
    errors, _ = preflight.check_configuration("containers", source, selected, stacks)
    assert any("model.safetensors" in error and "Qwen" in error for error in errors)
    _asset(assets["qwen_weights"])
    shard.unlink()
    errors, _ = preflight.check_configuration("containers", source, selected, stacks)
    assert any("Qwen weight shard" in error for error in errors)


@pytest.mark.parametrize("asset_kind", ["shard", "tokenizer", "manifest", "script"])
def test_preflight_qwen_files_cannot_escape_the_model_directory_via_symlinks(
    preflight,
    deployment,
    tmp_path,
    monkeypatch,
    asset_kind,
) -> None:
    source, values, _, assets = deployment
    qwen = assets["qwen_weights"].parent
    outside = _asset(tmp_path / "outside.safetensors")
    if asset_kind == "tokenizer":
        assets["qwen_tokenizer"].unlink()
        assets["qwen_tokenizer"].symlink_to(outside)
    elif asset_kind == "script":
        assets["qwen_script"].unlink()
        assets["qwen_script"].symlink_to(outside)
    elif asset_kind == "manifest":
        outside.write_text(json.dumps({"weight_map": {"a": "model.safetensors"}}), encoding="utf-8")
        (qwen / "model.safetensors.index.json").symlink_to(outside)
    else:
        shard = qwen / "model-00001-of-00001.safetensors"
        shard.symlink_to(outside)
        _asset(
            qwen / "model.safetensors.index.json",
            json.dumps({"weight_map": {"a": shard.name}}).encode(),
        )
    stacks = {"base", "embedding"}
    if asset_kind == "manifest":
        original_read = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == qwen / "model.safetensors.index.json":
                pytest.fail("preflight read the escaped manifest")
            return original_read(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)

    errors, _ = preflight.check_configuration("containers", source, _select(values, stacks), stacks)

    assert errors


@pytest.mark.parametrize("key", ["MODEL_DIR", "MEDIA_DIR", "CACHE_DIR", "QWEN_EMBEDDING_MODEL_DIR"])
def test_preflight_rejects_relative_deployment_directories(preflight, deployment, key) -> None:
    source, values, _, _ = deployment
    values = _select(values, {"base", "embedding"})
    values[key] = "relative-directory"

    errors, _ = preflight.check_configuration("containers", source, values, {"base", "embedding"})

    assert f"{key} must be an absolute path" in errors


@pytest.mark.parametrize("key", ["MODEL_DIR", "MEDIA_DIR", "CACHE_DIR", "QWEN_EMBEDDING_MODEL_DIR"])
@pytest.mark.parametrize("invalid_kind", ["file", "host-root"])
def test_preflight_storage_paths_are_dedicated_directories(
    preflight,
    deployment,
    tmp_path,
    key,
    invalid_kind,
) -> None:
    source, values, _, _ = deployment
    values = _select(values, {"base", "embedding"})
    values[key] = str(_asset(tmp_path / "not-a-directory")) if invalid_kind == "file" else "/"

    errors, _ = preflight.check_configuration("containers", source, values, {"base", "embedding"})

    assert any(key in error and "directory" in error for error in errors)


def test_preflight_rejects_enabling_automatic_model_downloads(preflight, deployment) -> None:
    source, values, _, _ = deployment
    values["FACE_INSIGHTFACE_ALLOW_DOWNLOAD"] = "true"

    errors, _ = preflight.check_configuration("containers", source, values, {"base"})

    assert "automatic face-model downloads must remain disabled" in errors


def test_preflight_missing_models_never_download_or_run_commands(
    preflight,
    deployment,
    monkeypatch,
) -> None:
    source, values, _, assets = deployment
    assets["face_detector"].unlink()
    assets["qwen_weights"].unlink()
    before = _snapshot(source.parent)

    def unexpected(*args, **kwargs):
        pytest.fail("configuration checks attempted a network operation or external command")

    monkeypatch.setattr(socket, "create_connection", unexpected)
    monkeypatch.setattr(request, "urlopen", unexpected)
    monkeypatch.setattr(preflight.subprocess, "run", unexpected)
    stacks = {"base", "reid", "embedding"}

    errors, _ = preflight.check_configuration("containers", source, _select(values, stacks), stacks)

    assert errors
    assert _snapshot(source.parent) == before


@pytest.mark.parametrize(
    "stacks,overrides,message",
    [
        ({"reid"}, {"REID_ENABLED": "true"}, "stacks must include base"),
        ({"base", "unknown"}, {}, "unknown stack"),
        ({"base"}, {"REID_ENABLED": "true"}, "REID_ENABLED must agree"),
        ({"base", "semantic"}, {"SEMANTIC_SEARCH_ENABLED": "true"}, "semantic requires embedding"),
        ({"base"}, {"SEMANTIC_SEARCH_ENABLED": "true"}, "SEMANTIC_SEARCH_ENABLED must agree"),
    ],
)
def test_preflight_rejects_inconsistent_stack_configuration(
    preflight,
    deployment,
    stacks,
    overrides,
    message,
) -> None:
    source, values, _, _ = deployment

    errors, _ = preflight.check_configuration("containers", source, {**values, **overrides}, stacks)

    assert any(message in error for error in errors)


@pytest.mark.parametrize("model_name", ["../outside", "/outside", ".", ".."])
def test_preflight_face_model_name_cannot_override_or_escape_its_root(
    preflight,
    deployment,
    model_name,
) -> None:
    source, values, _, _ = deployment
    values = _select(values, {"base", "reid"})
    values["FACE_INSIGHTFACE_MODEL"] = model_name

    errors, _ = preflight.check_configuration("containers", source, values, {"base", "reid"})

    assert any("FACE_INSIGHTFACE_MODEL" in error for error in errors)
    assert all(model_name not in error for error in errors)


@pytest.mark.parametrize("asset_name", ["yolo", "qwen_weights", "qwen_tokenizer", "qwen_script"])
def test_preflight_requires_readable_assets(preflight, deployment, monkeypatch, asset_name) -> None:
    source, values, _, assets = deployment
    unreadable = assets[asset_name]
    original_access = preflight.os.access
    monkeypatch.setattr(
        preflight.os,
        "access",
        lambda path, mode: False if Path(path) == unreadable else original_access(path, mode),
    )
    stacks = {"base", "embedding"}

    errors, _ = preflight.check_configuration("containers", source, _select(values, stacks), stacks)

    assert errors


def test_preflight_tools_only_run_readonly_inspection_commands(preflight, monkeypatch) -> None:
    calls = []
    outputs = {
        "docker": "linux",
        "node": "v22.18.0",
        "nvidia-smi": "NVIDIA GeForce RTX 5090, 12000",
    }

    def run(arguments, **kwargs):
        calls.append(arguments)
        assert kwargs == {"check": False, "capture_output": True, "text": True, "timeout": 15}
        return subprocess.CompletedProcess(arguments, 0, outputs[arguments[0]], "")

    monkeypatch.setattr(preflight.shutil, "which", lambda tool: f"/synthetic/bin/{tool}")
    monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
    monkeypatch.setattr(preflight.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(preflight.subprocess, "run", run)

    assert preflight.check_tools("rtx5090", {"base", "reid"}) == []
    assert calls == [
        ["docker", "compose", "version"],
        ["docker", "info", "--format", "{{.OSType}}"],
        ["node", "--version"],
        ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"],
    ]


@pytest.mark.parametrize(
    "node,gpu,system,machine,message",
    [
        ("v22.17.0", "RTX 5090, 12000", "Linux", "x86_64", "Node.js 22.18"),
        ("v22.18.0", "RTX 5090, 4000", "Linux", "x86_64", "at least 4096 MiB"),
        ("v22.18.0", "RTX 5090, unknown", "Linux", "x86_64", "at least 4096 MiB"),
        ("v22.18.0", "RTX 4090, 12000", "Linux", "x86_64", "RTX 5090 on GPU 0"),
        ("v22.18.0", "RTX 5090, 12000", "Darwin", "arm64", "Linux x86_64"),
    ],
)
def test_preflight_rtx_tools_reject_unsupported_runtime_or_capacity(
    preflight,
    monkeypatch,
    node,
    gpu,
    system,
    machine,
    message,
) -> None:
    def run(arguments, **kwargs):
        output = (
            node if arguments[0] == "node" else gpu if arguments[0] == "nvidia-smi" else "linux"
        )
        return subprocess.CompletedProcess(arguments, 0, output, "")

    monkeypatch.setattr(preflight.shutil, "which", lambda tool: f"/synthetic/bin/{tool}")
    monkeypatch.setattr(preflight.platform, "system", lambda: system)
    monkeypatch.setattr(preflight.platform, "machine", lambda: machine)
    monkeypatch.setattr(preflight.subprocess, "run", run)

    errors = preflight.check_tools("rtx5090", {"base", "reid"})

    assert any(message in error for error in errors)


def test_preflight_requires_a_linux_docker_engine(preflight, monkeypatch) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda tool: f"/synthetic/bin/{tool}")
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda arguments, **kwargs: subprocess.CompletedProcess(arguments, 0, "windows", ""),
    )

    assert "Docker must use Linux containers" in preflight.check_tools("containers", {"base"})


def test_preflight_missing_tools_produce_diagnostics_without_commands(
    preflight, monkeypatch
) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda tool: None)
    monkeypatch.setattr(
        preflight.subprocess, "run", lambda *a, **k: pytest.fail("unexpected command")
    )

    errors = preflight.check_tools("containers", {"base"})

    assert errors == [f"required tool missing: {tool}" for tool in ("docker", "git", "tar")]


@pytest.mark.parametrize("failure", ["exit", "timeout", "oserror"])
def test_preflight_tool_errors_do_not_echo_secret_command_output(
    preflight,
    deployment,
    monkeypatch,
    capsys,
    failure,
) -> None:
    source, _, env_file, _ = deployment
    monkeypatch.setattr(preflight.shutil, "which", lambda tool: f"/synthetic/bin/{tool}")

    def run(arguments, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(arguments, 15, output=SECRETS[0])
        if failure == "oserror":
            raise OSError(SECRETS[0])
        return subprocess.CompletedProcess(arguments, 1, SECRETS[0], SECRETS[1])

    monkeypatch.setattr(preflight.subprocess, "run", run)

    assert (
        preflight.main(["--source", str(source), "--env-file", str(env_file), "--check-tools"]) == 1
    )
    output = capsys.readouterr()
    assert "read-only tool check failed: docker" in output.err
    assert all(secret not in output.out + output.err for secret in SECRETS)


@pytest.mark.parametrize(
    "invalid_line",
    [
        f"APP_BASIC_AUTH_PASSWORD='{SECRETS[0]}",
        f"APP_BASIC_AUTH_PASSWORD={SECRETS[0]};touch",
    ],
)
def test_preflight_parse_errors_redact_secret_values(deployment, invalid_line) -> None:
    source, _, env_file, _ = deployment
    env_file.write_text(invalid_line + "\n", encoding="utf-8")

    result = _cli(source, env_file, {"base"})

    assert result.returncode == 1
    assert "APP_BASIC_AUTH_PASSWORD" in result.stderr
    assert "Traceback" not in result.stderr
    assert all(secret not in result.stdout + result.stderr for secret in SECRETS)


@pytest.mark.parametrize(
    "key", ["APP_BASIC_AUTH_PASSWORD", "REID_SERVICE_API_KEY", "QWEN_EMBEDDING_API_KEY"]
)
def test_preflight_cli_cannot_query_secret_values(deployment, key) -> None:
    source, _, env_file, _ = deployment
    result = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--source",
            str(source),
            "--env-file",
            str(env_file),
            "--value",
            key,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert all(secret not in result.stdout + result.stderr for secret in SECRETS)
