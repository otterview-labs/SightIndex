"""Exercise container orchestration against a temporary fake Docker CLI only."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANAGE = ROOT / "deploy/containers/manage.sh"
API_CONTAINER = "a" * 64
REID_CONTAINER = "d" * 64
EMBEDDING_CONTAINER = "e" * 64
IMAGE_ID = "sha256:" + "b" * 64
EMBEDDING_IMAGE_ID = "sha256:" + "f" * 64


@pytest.fixture
def deployment(tmp_path):
    release = tmp_path / "releases/current"
    compose_dir = release / "deploy/containers"
    compose_dir.mkdir(parents=True)
    for name in ("compose.yaml", "compose.embedding.yaml", "compose.semantic-search.yaml"):
        shutil.copyfile(ROOT / "deploy/containers" / name, compose_dir / name)
    env_file = tmp_path / "synthetic.env"
    env_file.write_text(
        "SIGHTINDEX_IMAGE=sightindex:unit-release\n"
        "REID_ENABLED=false\n"
        "QWEN_EMBEDDING_IMAGE=sightindex-embedding:unit-release\n"
        "QWEN_EMBEDDING_MODEL_DIR=/unused/unit-model\n"
        "QWEN_EMBEDDING_API_KEY=synthetic-not-a-secret\n"
    )
    calls_file = tmp_path / "docker-calls.jsonl"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_cli = (
        f"#!{sys.executable}\n"
        + """\
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
cli = Path(sys.argv[0]).name
with open(os.environ["FAKE_DOCKER_CALLS"], "a", encoding="utf-8") as record:
    record.write(json.dumps({"cli": cli, "args": args}) + "\\n")
if cli == "docker" and args == ["compose", "version"]:
    sys.exit(int(os.environ.get("FAKE_NO_COMPOSE", "0")))
if cli == "docker" and args[:2] == ["image", "inspect"]:
    variable = (
        "FAKE_EXPECTED_EMBEDDING_IMAGE_ID"
        if args[-1].startswith("sightindex-embedding:") else "FAKE_EXPECTED_IMAGE_ID"
    )
    print(os.environ[variable])
    sys.exit(0)
if cli == "docker" and args[:1] == ["inspect"]:
    if args[-1] == os.environ["FAKE_REID_CONTAINER"]:
        print(os.environ.get("FAKE_RUNNING_REID_IMAGE_ID", os.environ["FAKE_EXPECTED_IMAGE_ID"]))
    elif args[-1] == os.environ["FAKE_EMBEDDING_CONTAINER"]:
        print(os.environ.get(
            "FAKE_RUNNING_EMBEDDING_IMAGE_ID", os.environ["FAKE_EXPECTED_EMBEDDING_IMAGE_ID"]
        ))
    else:
        print(os.environ.get("FAKE_RUNNING_IMAGE_ID", os.environ["FAKE_EXPECTED_IMAGE_ID"]))
    sys.exit(0)
if cli == "docker":
    if args[:1] != ["compose"]:
        sys.exit(90)
    args = args[1:]
while args and args[0] in ("--project-name", "--env-file", "-f", "--profile"):
    args = args[2:]
if args == ["ls"]:
    if os.environ.get("FAKE_RUNNING_BASE"):
        print("sightindex-bj-test running " + os.environ["FAKE_RUNNING_BASE"])
elif args == ["ps", "--quiet", "api"]:
    print(os.environ.get("FAKE_API_CONTAINERS", os.environ["FAKE_API_CONTAINER"]))
elif args == ["ps", "--quiet", "reid"]:
    print(os.environ.get("FAKE_REID_CONTAINERS", os.environ["FAKE_REID_CONTAINER"]))
elif args == ["ps", "--quiet", "embedding"]:
    print(os.environ.get("FAKE_EMBEDDING_CONTAINERS", os.environ["FAKE_EMBEDDING_CONTAINER"]))
elif args[:1] == ["exec"]:
    print("Synthetic deployment verification")
    sys.exit(int(os.environ.get("FAKE_VERIFY_EXIT", "0")))
elif args[:1] not in (["up"], ["restart"], ["logs"], ["config"], ["down"], ["ps"]):
    sys.exit(91)
"""
    )
    for cli in ("docker", "docker-compose"):
        executable = fake_bin / cli
        executable.write_text(fake_cli)
        executable.chmod(0o755)

    def run(*arguments, docker_env=None, stacks_enabled=False, explicit_release=True):
        if stacks_enabled:
            env_file.write_text(
                env_file.read_text().replace("REID_ENABLED=false", "REID_ENABLED=true")
            )
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": str(fake_bin) + os.pathsep + environment.get("PATH", ""),
                "SIGHTINDEX_ROOT": str(tmp_path),
                "FAKE_DOCKER_CALLS": str(calls_file),
                "FAKE_EXPECTED_IMAGE_ID": IMAGE_ID,
                "FAKE_EXPECTED_EMBEDDING_IMAGE_ID": EMBEDDING_IMAGE_ID,
                "FAKE_API_CONTAINER": API_CONTAINER,
                "FAKE_REID_CONTAINER": REID_CONTAINER,
                "FAKE_EMBEDDING_CONTAINER": EMBEDDING_CONTAINER,
            }
        )
        environment.update(docker_env or {})
        command = ["bash", str(MANAGE), "--env-file", str(env_file)]
        if explicit_release:
            command.extend(["--release", str(release)])
        result = subprocess.run(
            [*command, *arguments],
            cwd=tmp_path,
            env=environment,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        calls = (
            [json.loads(line) for line in calls_file.read_text().splitlines()]
            if calls_file.exists()
            else []
        )
        return result, calls

    return run, release, env_file


def _action(calls, name):
    matches = [call["args"] for call in calls if name in call["args"]]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("arguments", [("logs", "api"), ("logs", "--service", "api")])
def test_manage_logs_supports_api_service_shorthand_and_flag(deployment, arguments):
    run, release, env_file = deployment
    result, calls = run(*arguments)
    assert result.returncode == 0, result.stderr
    invocation = _action(calls, "logs")
    assert invocation == [
        "compose",
        "--project-name",
        "sightindex-bj-test",
        "--env-file",
        str(env_file),
        "-f",
        str(release / "deploy/containers/compose.yaml"),
        "logs",
        "--tail=100",
        "-f",
        "api",
    ]


@pytest.mark.parametrize("command", ["up", "down", "config", "verify"])
def test_manage_scopes_every_operation_to_explicit_project(deployment, command):
    run, _release, _env_file = deployment
    result, calls = run("--project-name", "sightindex-isolated", command)
    assert result.returncode == 0, result.stderr
    project_calls = [
        call["args"]
        for call in calls
        if call["args"][:1] == ["compose"] and call["args"][1:] not in (["version"], ["ls"])
    ]
    assert project_calls
    assert all(args[1:3] == ["--project-name", "sightindex-isolated"] for args in project_calls)


@pytest.mark.parametrize("project", ["", "BadProject", "../escape", "x;echo", "x" * 64])
def test_manage_rejects_invalid_project_before_docker(deployment, project):
    run, _release, env_file = deployment
    env_file.write_text(env_file.read_text() + f"COMPOSE_PROJECT_NAME={project}\n")
    # An empty env value retains the backward-compatible default.
    if not project:
        result, _calls = run("config")
        assert result.returncode == 0
        return
    result, calls = run("config")
    assert result.returncode == 2
    assert calls == []


def test_manage_cannot_retarget_accepted_root_or_configured_project(deployment):
    run, release, env_file = deployment
    marker = release.parents[1] / "project-name"
    marker.write_text("sightindex-owned\n")
    result, calls = run("--project-name", "sightindex-other", "up")
    assert result.returncode == 1
    assert calls == []
    assert marker.read_text() == "sightindex-owned\n"
    env_file.write_text(env_file.read_text() + "COMPOSE_PROJECT_NAME=sightindex-other\n")
    result, calls = run("up")
    assert result.returncode == 1
    assert calls == []


@pytest.mark.parametrize("marker_value", ["", "first\nsecond\n", "UPPER\n"])
def test_manage_rejects_invalid_existing_project_marker(deployment, marker_value):
    run, release, _env_file = deployment
    (release.parents[1] / "project-name").write_text(marker_value)
    result, calls = run("--project-name", "sightindex-bj-test", "up")
    assert result.returncode == 1
    assert calls == []


def test_installed_manage_defaults_to_its_own_root(tmp_path):
    """No ambient variable is needed when invoking a copied ROOT/manage.sh."""
    installed = tmp_path / "independent"
    installed.mkdir()
    shutil.copyfile(MANAGE, installed / "manage.sh")
    # Invalid markers stop before Docker. The specific diagnostic proves that
    # the copied script read this root, not the legacy /data deployment.
    (installed / "project-name").write_text("")
    (installed / ".env").write_text("COMPOSE_PROJECT_NAME=sightindex-unit\n")
    environment = dict(os.environ)
    environment.pop("SIGHTINDEX_ROOT", None)
    result = subprocess.run(
        ["bash", str(installed / "manage.sh"), "up"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 1
    assert "Invalid accepted project name" in result.stderr


def test_manage_logs_accepts_only_allowlisted_service_arguments(deployment):
    run, _release, _env_file = deployment
    result, calls = run("logs", "--service", "--follow")
    assert result.returncode == 2
    assert "Unknown service" in result.stderr
    assert calls == []


def test_manage_rejects_service_selection_for_mutating_commands(deployment):
    run, _release, _env_file = deployment
    result, calls = run("up", "--service", "api")
    assert result.returncode == 2
    assert "only supported by logs" in result.stderr
    assert calls == []


def test_manage_default_up_selects_only_base_services(deployment):
    run, _release, _env_file = deployment
    result, calls = run("up")
    assert result.returncode == 0, result.stderr
    invocation = _action(calls, "up")
    assert invocation[-7:] == ["up", "-d", "postgres", "etcd", "minio", "milvus", "api"]
    assert "--profile" not in invocation
    assert not any("compose.embedding.yaml" in arg for arg in invocation)


def test_manage_explicit_stacks_preserve_compose_overlays_and_profiles(deployment):
    run, release, _env_file = deployment
    result, calls = run("up", "base", "reid", "embedding", "semantic", stacks_enabled=True)
    assert result.returncode == 0, result.stderr
    invocation = _action(calls, "up")
    for filename in ("compose.yaml", "compose.embedding.yaml", "compose.semantic-search.yaml"):
        assert str(release / "deploy/containers" / filename) in invocation
    assert invocation[-9:] == [
        "up",
        "-d",
        "postgres",
        "etcd",
        "minio",
        "milvus",
        "api",
        "reid",
        "embedding",
    ]
    assert invocation[invocation.index("--profile") :] == [
        "--profile",
        "reid",
        "--profile",
        "embedding",
        *invocation[-9:],
    ]


def test_manage_restart_does_not_claim_to_apply_new_image_or_config(deployment):
    run, _release, _env_file = deployment
    result, calls = run("restart", docker_env={"FAKE_RUNNING_BASE": "/old-release/compose.yaml"})
    assert result.returncode == 0, result.stderr
    assert _action(calls, "restart")[-6:] == [
        "restart",
        "postgres",
        "etcd",
        "minio",
        "milvus",
        "api",
    ]
    assert "'restart' only restarts existing containers" in result.stderr
    assert "'up'/'restart' will re-create" not in result.stderr
    assert not any("up" in call["args"] for call in calls)


def test_manage_config_is_quiet_and_does_not_start_services(deployment):
    run, _release, _env_file = deployment
    result, calls = run("config")
    assert result.returncode == 0, result.stderr
    assert _action(calls, "config")[-2:] == ["config", "--quiet"]
    assert not any("up" in call["args"] or "exec" in call["args"] for call in calls)
    assert "synthetic-not-a-secret" not in result.stdout + result.stderr


@pytest.mark.parametrize("model_flag", [(), ("--model-smoke",)])
def test_manage_verify_inspects_running_image_and_passes_selected_stacks(deployment, model_flag):
    run, _release, _env_file = deployment
    result, calls = run(
        "verify",
        "base",
        "reid",
        "embedding",
        "semantic",
        *model_flag,
        stacks_enabled=True,
    )
    assert result.returncode == 0, result.stderr
    assert ["image", "inspect", "--format", "{{.Id}}", "sightindex:unit-release"] in [
        call["args"] for call in calls
    ]
    assert ["inspect", "--format", "{{.Image}}", API_CONTAINER] in [call["args"] for call in calls]
    assert [call["args"][-3:] for call in calls if "ps" in call["args"]] == [
        ["ps", "--quiet", "api"],
        ["ps", "--quiet", "reid"],
        ["ps", "--quiet", "embedding"],
    ]
    assert ["image", "inspect", "--format", "{{.Id}}", "sightindex-embedding:unit-release"] in [
        call["args"] for call in calls
    ]
    assert ["inspect", "--format", "{{.Image}}", REID_CONTAINER] in [call["args"] for call in calls]
    assert ["inspect", "--format", "{{.Image}}", EMBEDDING_CONTAINER] in [
        call["args"] for call in calls
    ]
    invocation = _action(calls, "exec")
    assert invocation[invocation.index("exec") :] == [
        "exec",
        "-T",
        "api",
        "python",
        "/opt/sightindex/deployment_verify.py",
        "--stacks",
        "base reid embedding semantic",
        "--upload-smoke",
        *model_flag,
    ]
    assert "API image verified: " + IMAGE_ID in result.stdout
    assert "ReID image verified: " + IMAGE_ID in result.stdout
    assert "Embedding image verified: " + EMBEDDING_IMAGE_ID in result.stdout


def test_manage_verify_rejects_old_image_under_the_same_tag(deployment):
    run, _release, _env_file = deployment
    result, calls = run("verify", docker_env={"FAKE_RUNNING_IMAGE_ID": "sha256:" + "c" * 64})
    assert result.returncode == 1
    assert "API image mismatch" in result.stderr
    assert not any("exec" in call["args"] for call in calls)


@pytest.mark.parametrize(
    ("stack", "variable", "label"),
    [
        ("reid", "FAKE_RUNNING_REID_IMAGE_ID", "ReID"),
        ("embedding", "FAKE_RUNNING_EMBEDDING_IMAGE_ID", "Embedding"),
    ],
)
def test_manage_verify_rejects_stale_selected_model_service(deployment, stack, variable, label):
    run, _release, _env_file = deployment
    result, calls = run(
        "verify",
        stack,
        docker_env={variable: "sha256:" + "c" * 64},
        stacks_enabled=True,
    )
    assert result.returncode == 1
    assert label + " image mismatch" in result.stderr
    assert not any("exec" in call["args"] for call in calls)
    assert not any("up" in call["args"] or "restart" in call["args"] for call in calls)


@pytest.mark.parametrize(
    ("stack", "variable", "label", "container"),
    [
        ("reid", "FAKE_REID_CONTAINERS", "ReID", REID_CONTAINER),
        ("embedding", "FAKE_EMBEDDING_CONTAINERS", "Embedding", EMBEDDING_CONTAINER),
    ],
)
@pytest.mark.parametrize("multiple", [False, True])
def test_manage_verify_requires_exactly_one_selected_model_container(
    deployment, stack, variable, label, container, multiple
):
    run, _release, _env_file = deployment
    value = container + "\n" + "c" * 64 if multiple else ""
    result, calls = run("verify", stack, docker_env={variable: value}, stacks_enabled=True)
    assert result.returncode == 1
    assert "one running " + label + " container" in result.stderr
    assert not any("exec" in call["args"] for call in calls)


def test_manage_base_verification_does_not_require_unselected_models(deployment):
    run, _release, _env_file = deployment
    result, calls = run(
        "verify", docker_env={"FAKE_REID_CONTAINERS": "", "FAKE_EMBEDDING_CONTAINERS": ""}
    )
    assert result.returncode == 0, result.stderr
    assert [call["args"][-3:] for call in calls if "ps" in call["args"]] == [
        ["ps", "--quiet", "api"]
    ]


@pytest.mark.parametrize("containers", ["", API_CONTAINER + "\n" + "d" * 64])
def test_manage_verify_requires_exactly_one_running_api_container(deployment, containers):
    run, _release, _env_file = deployment
    result, calls = run("verify", docker_env={"FAKE_API_CONTAINERS": containers})
    assert result.returncode == 1
    assert "one running API container" in result.stderr
    assert not any("exec" in call["args"] or "inspect" in call["args"] for call in calls)


def test_manage_verify_propagates_acceptance_helper_failure(deployment):
    run, _release, _env_file = deployment
    result, calls = run("verify", docker_env={"FAKE_VERIFY_EXIT": "7"})
    assert result.returncode == 7
    assert _action(calls, "exec")[-3:] == ["--stacks", "base", "--upload-smoke"]


def test_manage_config_supports_the_standalone_compose_cli(deployment):
    run, _release, _env_file = deployment
    result, calls = run("config", docker_env={"FAKE_NO_COMPOSE": "1"})
    assert result.returncode == 0, result.stderr
    assert next(call for call in calls if "config" in call["args"])["cli"] == "docker-compose"


def test_manage_down_keeps_volumes_and_accounts_for_optional_stacks(deployment):
    run, _release, _env_file = deployment
    result, calls = run("down")
    assert result.returncode == 0, result.stderr
    invocation = _action(calls, "down")
    assert invocation[-5:] == ["--profile", "reid", "--profile", "embedding", "down"]
    assert "-v" not in invocation
    assert "--volumes" not in invocation


def test_manage_prefers_accepted_active_release_over_newer_staged_directory(deployment):
    run, release, env_file = deployment
    pending_release = env_file.parent / "releases/unverified-newest"
    pending_compose = pending_release / "deploy/containers/compose.yaml"
    pending_compose.parent.mkdir(parents=True)
    shutil.copyfile(release / "deploy/containers/compose.yaml", pending_compose)
    os.utime(pending_release, (2_000_000_000, 2_000_000_000))
    (env_file.parent / "active-release").write_text(str(release.resolve()) + "\n")
    result, calls = run("config", explicit_release=False)
    assert result.returncode == 0, result.stderr
    invocation = _action(calls, "config")
    assert str(release.resolve() / "deploy/containers/compose.yaml") in invocation
    assert str(pending_compose) not in invocation


def test_manage_explicit_release_overrides_acceptance_marker(deployment):
    run, release, env_file = deployment
    (env_file.parent / "active-release").write_text("/nonexistent/old-release\n")
    result, calls = run("config")
    assert result.returncode == 0, result.stderr
    assert str(release / "deploy/containers/compose.yaml") in _action(calls, "config")


def test_manage_without_acceptance_marker_keeps_legacy_release_lookup(deployment):
    run, release, _env_file = deployment
    result, calls = run("config", explicit_release=False)
    assert result.returncode == 0, result.stderr
    assert str(release / "deploy/containers/compose.yaml") in _action(calls, "config")


@pytest.mark.parametrize("marker", ["relative-release\n", "\n", "/abs/path\n/second/path\n"])
def test_manage_rejects_invalid_active_release_marker_before_docker(deployment, marker):
    run, _release, env_file = deployment
    (env_file.parent / "active-release").write_text(marker)
    result, calls = run("config", explicit_release=False)
    assert result.returncode == 1
    assert "Invalid active-release marker" in result.stderr
    assert calls == []


def test_manage_rejects_active_release_symlink_escape(deployment):
    run, _release, env_file = deployment
    outside = env_file.parent / "outside-releases"
    outside.mkdir()
    linked_release = env_file.parent / "releases/escaped"
    linked_release.symlink_to(outside, target_is_directory=True)
    (env_file.parent / "active-release").write_text(str(linked_release) + "\n")
    result, calls = run("config", explicit_release=False)
    assert result.returncode == 1
    assert "release must be inside" in result.stderr
    assert calls == []


def test_manage_rejects_symlinked_active_release_marker(deployment):
    run, release, env_file = deployment
    other_marker = env_file.parent / "other-marker"
    other_marker.write_text(str(release) + "\n")
    (env_file.parent / "active-release").symlink_to(other_marker)
    result, calls = run("config", explicit_release=False)
    assert result.returncode == 1
    assert "expected a regular file" in result.stderr
    assert calls == []


def test_manage_legacy_lookup_skips_pending_deployments(deployment):
    run, release, env_file = deployment
    pending_release = env_file.parent / "releases/pending-newest"
    pending_compose = pending_release / "deploy/containers/compose.yaml"
    pending_compose.parent.mkdir(parents=True)
    shutil.copyfile(release / "deploy/containers/compose.yaml", pending_compose)
    (pending_release / ".deployment-pending").touch()
    os.utime(pending_release, (2_000_000_000, 2_000_000_000))
    result, calls = run("config", explicit_release=False)
    assert result.returncode == 0, result.stderr
    invocation = _action(calls, "config")
    assert str(release / "deploy/containers/compose.yaml") in invocation
    assert str(pending_compose) not in invocation


@pytest.mark.parametrize(
    "image_value",
    [
        '"sightindex:unit-release"',
        "'sightindex:unit-release' # release comment",
        "sightindex:unit-release # release comment",
    ],
)
def test_manage_verify_accepts_quoted_image_settings_without_shell_evaluation(
    deployment,
    image_value,
):
    run, _release, env_file = deployment
    env_file.write_text(
        env_file.read_text().replace("SIGHTINDEX_IMAGE=sightindex:unit-release", "")
        + "export SIGHTINDEX_IMAGE = "
        + image_value
        + "\r\n"
    )
    result, calls = run("verify")
    assert result.returncode == 0, result.stderr
    assert ["image", "inspect", "--format", "{{.Id}}", "sightindex:unit-release"] in [
        call["args"] for call in calls
    ]


def test_manage_split_overlay_lookup_does_not_use_another_pending_release(deployment):
    run, release, env_file = deployment
    overlay = release / "deploy/containers/compose.embedding.yaml"
    accepted_overlay = (
        env_file.parent / "releases/accepted-overlay/deploy/containers" / overlay.name
    )
    pending_overlay = env_file.parent / "releases/pending-overlay/deploy/containers" / overlay.name
    for destination in (accepted_overlay, pending_overlay):
        destination.parent.mkdir(parents=True)
        shutil.copyfile(overlay, destination)
    pending_release = pending_overlay.parents[2]
    (pending_release / ".deployment-pending").touch()
    os.utime(pending_release, (2_000_000_000, 2_000_000_000))
    overlay.unlink()
    result, calls = run("config", "embedding")
    assert result.returncode == 0, result.stderr
    invocation = _action(calls, "config")
    assert str(accepted_overlay) in invocation
    assert str(pending_overlay) not in invocation


def test_manage_explicit_release_can_validate_a_pending_deployment(deployment):
    run, release, _env_file = deployment
    (release / ".deployment-pending").touch()
    result, calls = run("config", "embedding")
    assert result.returncode == 0, result.stderr
    assert str(release / "deploy/containers/compose.embedding.yaml") in _action(calls, "config")
