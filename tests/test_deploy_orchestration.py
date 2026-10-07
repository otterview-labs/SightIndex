"""Run real deployment Bash with fake tools and temporary source/config/model assets.

No test invokes Docker, Git, a GPU runtime, or a remote service. Preflight and
manage.sh are real; the external command boundary is replaced by a strict recorder.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
REVISION = "abc123456789012345678901234567890123456789"
CANDIDATE_ID = "sha256:" + "a" * 64
PREVIOUS_ID = "sha256:" + "b" * 64

MOCK_TOOL = """\
import json
import os
import runpy
import sys
import time
from pathlib import Path

tool = Path(sys.argv[0]).name
args = sys.argv[1:]
with Path(os.environ["ORCHESTRATION_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"tool": tool, "args": args}) + "\\n")

def flag(name):
    return os.environ.get(name) == "1"

def image_id(tag):
    return "sha256:" + ("b" if tag == "sightindex:old" else "a") * 64

if tool == "fake-python":
    if args and args[0] == "-":
        # Execute the actual archive manifest against the temporary checkout.
        script = sys.stdin.read()
        if "SNAPSHOT_MANIFEST.json" in script:
            assert Path(args[1]).is_relative_to(Path(os.environ["ORCHESTRATION_ROOT"]) / "releases")
        else:
            assert "blocked_directories" in script and "for root in sys.argv[1:]" in script
        sys.argv = args
        exec(compile(script, "<deployment-snapshot>", "exec"), {"__name__": "__main__"})
    elif args and args[0].endswith("preflight.py"):
        sys.argv = args
        runpy.run_path(args[0], run_name="__main__")
    else:
        raise SystemExit("unexpected Python operation in deployment test")
elif tool == "git":
    if args[0] != "-C":
        raise SystemExit("unscoped Git operation")
    if args[2:] == ["rev-parse", "HEAD"]:
        print("abc123456789012345678901234567890123456789")
    elif args[2:] == ["status", "--porcelain", "--untracked-files=normal"]:
        if flag("MOCK_DIRTY"):
            print(" M app/example.py")
    else:
        raise SystemExit("unexpected Git operation")
elif tool == "sleep":
    # Let Bash SECONDS advance while keeping the one-second timeout test quick.
    time.sleep(0.01)
elif tool == "id":
    assert args == ["-u"]
    print(os.environ.get("MOCK_UID", "1000"))
elif tool == "chown":
    assert len(args) == 2 and args[0] == "1000:1000"
    assert Path(args[1]).is_relative_to(Path(os.environ["ORCHESTRATION_ENV"]).parent)
    raise SystemExit(1 if flag("MOCK_CHOWN_FAIL") else 0)
elif tool == "nvidia-smi":
    if args != ["--query-gpu=name,memory.free", "--format=csv,noheader,nounits"]:
        raise SystemExit("unexpected GPU operation")
    print("Synthetic RTX, 24576")
elif tool == "docker":
    if args == ["info", "--format", "{{.OSType}}"]:
        print("linux")
    elif args[:2] == ["volume", "inspect"]:
        raise SystemExit(0 if flag("MOCK_VOLUME_EXISTS") else 1)
    elif args[:2] == ["image", "inspect"]:
        if flag("MOCK_IMAGE_MISSING"):
            raise SystemExit(1)
        if "--format" in args:
            print(image_id(args[-1]))
        else:
            print("[]")
    elif args and args[0] == "build":
        if flag("MOCK_BUILD_FAIL"):
            raise SystemExit(39)
        assert "--build-arg" in args and "-t" in args
        print("synthetic build completed")
    elif args and args[0] == "inspect":
        if args[2] == "{{.Image}}":
            if flag("MOCK_IMAGE_MISMATCH"):
                print("sha256:" + "c" * 64)
            else:
                env_path = Path(os.environ["ORCHESTRATION_ENV"])
                image_key = "QWEN_EMBEDDING_IMAGE" if args[-1] == "ff" * 6 else "SIGHTINDEX_IMAGE"
                image = next(line.split("=", 1)[1] for line in env_path.read_text().splitlines()
                             if line.startswith(image_key + "="))
                print(image_id(image))
        elif "State.Status" in args[2]:
            print("running starting" if flag("MOCK_UNHEALTHY") else "running healthy")
        else:
            raise SystemExit("unexpected Docker inspect format")
    elif args and args[0] == "compose":
        index = 1
        options = {"--project-name", "--env-file", "-f", "--profile"}
        while index < len(args) and args[index] in options:
            index += 2
        operation = args[index] if index < len(args) else ""
        rest = args[index + 1:]
        if operation == "version":
            print("Docker Compose version v2.mock")
        elif operation == "ls":
            print("NAME STATUS CONFIG FILES")
        elif operation == "ps":
            if "--all" in rest:
                if flag("MOCK_PREVIOUS_DATABASE"):
                    print("dd" * 6)
            elif rest and rest[-1] == "api":
                print("aa" * 6)
            elif rest and rest[-1] == "embedding":
                print("ff" * 6)
            elif rest and rest[-1] in {"postgres", "etcd", "minio", "milvus", "reid"}:
                print("ee" * 6)
            else:
                raise SystemExit("unexpected Compose ps operation")
        elif operation == "exec" and "pg_dump" in rest:
            if flag("MOCK_BACKUP_FAIL"):
                print("synthetic private backup failure", file=sys.stderr)
                raise SystemExit(47)
            print("PGDUMP-synthetic-fixture")
        elif operation == "exec" and "/opt/sightindex/deployment_verify.py" in rest:
            assert rest[:4] == ["-T", "api", "python", "/opt/sightindex/deployment_verify.py"]
            assert "--upload-smoke" in rest and "--stacks" in rest
            raise SystemExit(53 if flag("MOCK_VERIFY_FAIL") else 0)
        elif operation == "config":
            assert rest == ["--quiet"]
        elif operation == "up":
            assert rest[0] == "-d" and "api" in rest
            if flag("MOCK_UP_FAIL"):
                raise SystemExit(59)
        else:
            raise SystemExit("unexpected Compose operation: " + operation)
    else:
        raise SystemExit("unexpected Docker operation")
else:
    raise SystemExit("unexpected test tool")
"""


@dataclass(frozen=True)
class Deployment:
    """Temporary boundary for one real Bash deployment invocation."""

    source: Path
    root: Path
    env_file: Path
    tools: Path
    log: Path
    models: Path

    def run(
        self, *arguments: str, flags: Mapping[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run the public entry point with isolated configuration and mock commands."""
        environment = {
            "PATH": f"{self.tools}:/usr/bin:/bin:/usr/sbin:/sbin",
            "SIGHTINDEX_ROOT": str(self.root),
            "SIGHTINDEX_DEPLOY_PYTHON": str(self.tools / "fake-python"),
            "SIGHTINDEX_DEPLOY_TIMEOUT_SECONDS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "ORCHESTRATION_LOG": str(self.log),
            "ORCHESTRATION_ENV": str(self.env_file),
            "ORCHESTRATION_ROOT": str(self.root),
        }
        environment.update(flags or {})
        return subprocess.run(
            [
                "/bin/bash",
                str(self.source / "deploy.sh"),
                "--root",
                str(self.root),
                "--source",
                str(self.source),
                "--env-file",
                str(self.env_file),
                "--release",
                "candidate",
                *arguments,
            ],
            cwd=self.source,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=12,
        )

    def commands(self) -> list[dict[str, Any]]:
        """Read only the strict fake command recorder."""
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def previous_release(self) -> Path:
        """Seed an accepted prior release and active markers, without a real database."""
        previous = self.root / "releases" / "previous"
        (previous / "deploy" / "containers").mkdir(parents=True)
        shutil.copyfile(
            self.source / "deploy" / "containers" / "compose.yaml",
            previous / "deploy" / "containers" / "compose.yaml",
        )
        (self.root / "active-release").write_text(f"{previous}\n", encoding="utf-8")
        (self.root / "active-stacks").write_text("base\n", encoding="utf-8")
        return previous


@pytest.fixture
def deployment(tmp_path: Path) -> Deployment:
    """Copy scripts of record; all model bytes and private config belong to the fixture."""
    source = tmp_path / "source"
    containers = source / "deploy" / "containers"
    containers.mkdir(parents=True)
    for name in (
        "deploy.sh",
        "manage.sh",
        "preflight.py",
        "verify.py",
        "Dockerfile",
        "compose.yaml",
        "compose.embedding.yaml",
        "compose.semantic-search.yaml",
        ".env.example",
    ):
        shutil.copyfile(REPOSITORY / "deploy" / "containers" / name, containers / name)
    shutil.copyfile(REPOSITORY / "deploy.sh", source / "deploy.sh")
    (source / "deploy" / "rtx5090").mkdir()
    shutil.copyfile(
        REPOSITORY / "deploy" / "rtx5090" / "install_or_update.sh",
        source / "deploy" / "rtx5090" / "install_or_update.sh",
    )
    shutil.copyfile(REPOSITORY / ".dockerignore", source / ".dockerignore")
    (source / "app").mkdir()
    (source / "app" / "__init__.py").write_text("# synthetic source\n", encoding="utf-8")
    (source / "main.py").write_text("# synthetic source\n", encoding="utf-8")
    (source / "pyproject.toml").write_text('[project]\nname="synthetic"\n', encoding="utf-8")
    (source / "frontend").mkdir()
    (source / "frontend" / "package.json").write_text('{"private":true}\n', encoding="utf-8")
    models = tmp_path / "assets" / "models"
    face = models / "insightface" / "models" / "buffalo_l"
    face.mkdir(parents=True)
    for path in (models / "yolo11n.pt", face / "det_10g.onnx", face / "w600k_r50.onnx"):
        path.write_bytes(b"synthetic model fixture; not usable weights")
    values = dict(
        line.split("=", 1)
        for line in (containers / ".env.example").read_text().splitlines()
        if line and not line.startswith("#")
    )
    values.update(
        SIGHTINDEX_IMAGE="sightindex:old",
        MODEL_DIR=str(models),
        MEDIA_DIR=str(tmp_path / "assets" / "media"),
        CACHE_DIR=str(tmp_path / "assets" / "cache"),
        APP_BASIC_AUTH_PASSWORD="fixture_private_app_password",
        POSTGRES_PASSWORD="fixture_private_database_password",
        MINIO_ROOT_PASSWORD="fixture_private_minio_password",
        REID_SERVICE_API_KEY="fixture_private_service_key",
    )
    env_file = tmp_path / "private.env"
    env_file.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8"
    )
    env_file.chmod(0o600)
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("fake-python", "git", "docker", "sleep", "nvidia-smi", "id", "chown"):
        executable = tools / name
        executable.write_text(f"#!{sys.executable}\n" + MOCK_TOOL, encoding="utf-8")
        executable.chmod(0o700)
    return Deployment(
        source, tmp_path / "deployment", env_file, tools, tmp_path / "commands.jsonl", models
    )


def _docker_commands(deployment: Deployment) -> list[list[str]]:
    return [entry["args"] for entry in deployment.commands() if entry["tool"] == "docker"]


def _operations(deployment: Deployment, operation: str) -> list[list[str]]:
    return [arguments for arguments in _docker_commands(deployment) if operation in arguments]


def test_independent_project_scopes_backup_start_verify_and_marker(deployment: Deployment) -> None:
    """A custom project never inspects the default project's volume or container."""
    deployment.env_file.write_text(
        deployment.env_file.read_text().replace(
            "COMPOSE_PROJECT_NAME=sightindex-bj-test", "COMPOSE_PROJECT_NAME=sightindex-isolated"
        )
    )
    deployment.previous_release()
    result = deployment.run(
        "--project-name", "sightindex-isolated", flags={"MOCK_PREVIOUS_DATABASE": "1"}
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = _docker_commands(deployment)
    selected = [
        args
        for args in commands
        if args[:1] == ["compose"] and args != ["compose", "version"] and args != ["compose", "ls"]
    ]
    assert selected
    assert all(args[1:3] == ["--project-name", "sightindex-isolated"] for args in selected)
    assert (deployment.root / "project-name").read_text().strip() == "sightindex-isolated"
    assert "SIGHTINDEX_IMAGE=sightindex-isolated:candidate" in deployment.env_file.read_text()
    assert not any("sightindex-bj-test_postgres-data" in args for args in commands)


def test_custom_project_uses_only_its_own_unbacked_volume(deployment: Deployment) -> None:
    deployment.env_file.write_text(
        deployment.env_file.read_text().replace(
            "COMPOSE_PROJECT_NAME=sightindex-bj-test", "COMPOSE_PROJECT_NAME=sightindex-isolated"
        )
    )
    result = deployment.run(flags={"MOCK_VOLUME_EXISTS": "1"})
    assert result.returncode != 0
    assert ["volume", "inspect", "sightindex-isolated_postgres-data"] in _docker_commands(
        deployment
    )
    assert not _operations(deployment, "build")


@pytest.mark.parametrize("project", ["BadProject", "../escape", "x;echo", "x" * 64])
def test_invalid_or_conflicting_project_creates_nothing(
    deployment: Deployment, project: str
) -> None:
    before = deployment.env_file.read_bytes()
    result = deployment.run("--project-name", project)
    assert result.returncode != 0
    assert not deployment.root.exists()
    assert deployment.env_file.read_bytes() == before
    assert not _docker_commands(deployment)


def test_accepted_root_cannot_be_rebound_to_other_project(deployment: Deployment) -> None:
    deployment.root.mkdir()
    marker = deployment.root / "project-name"
    marker.write_text("sightindex-owned\n")
    result = deployment.run()
    assert result.returncode != 0
    assert marker.read_text() == "sightindex-owned\n"
    assert not _docker_commands(deployment)


def test_cli_root_is_forwarded_to_manage_despite_ambient_root(deployment: Deployment) -> None:
    """The --root flag governs child manage calls and their accepted markers."""
    other = deployment.root.parent / "other-deployment"
    other.mkdir()
    (other / "project-name").write_text("another-project\n")
    result = deployment.run(
        flags={"SIGHTINDEX_ROOT": str(other), "COMPOSE_PROJECT_NAME": "ambient-wrong"}
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (other / "project-name").read_text() == "another-project\n"
    assert not (other / "active-release").exists()
    assert (deployment.root / "project-name").read_text().strip() == "sightindex-bj-test"
    for command in _docker_commands(deployment):
        if command[:1] == ["compose"] and "--project-name" in command:
            assert command[command.index("--project-name") + 1] == "sightindex-bj-test"


def test_snapshot_keeps_orm_code_without_admitting_runtime_models(deployment: Deployment) -> None:
    source = deployment.source / "app" / "models"
    source.mkdir()
    (source / "media.py").write_text("# synthetic ORM model\n")
    (source / "private.pth").write_bytes(b"synthetic weight excluded")
    result = deployment.run()
    assert result.returncode == 0, result.stdout + result.stderr
    target = deployment.root / "releases" / "candidate" / "app" / "models"
    assert (target / "media.py").read_text() == "# synthetic ORM model\n"
    assert not (target / "private.pth").exists()


@pytest.mark.parametrize("arguments", [("--help",), ("--unknown",), ("--target", "unknown")])
def test_help_and_unknown_arguments_create_nothing(
    deployment: Deployment, arguments: tuple[str, ...]
) -> None:
    result = deployment.run(*arguments)
    assert result.returncode == (
        0 if arguments == ("--help",) else 1 if arguments == ("--unknown",) else 2
    )
    assert not deployment.root.exists()
    assert not deployment.commands()


def test_check_runs_real_preflight_with_only_readonly_mock_tools(deployment: Deployment) -> None:
    before = deployment.env_file.read_bytes()
    result = deployment.run("--check")
    assert result.returncode == 0, result.stderr
    assert "Read-only preflight passed" in result.stdout
    assert not deployment.root.exists()
    assert deployment.env_file.read_bytes() == before
    assert _docker_commands(deployment) == [
        ["compose", "version"],
        ["info", "--format", "{{.OSType}}"],
    ]
    assert not any(entry["tool"] == "git" for entry in deployment.commands())


def test_missing_model_fails_before_build_up_and_image_change(deployment: Deployment) -> None:
    before = deployment.env_file.read_bytes()
    (deployment.models / "yolo11n.pt").unlink()
    result = deployment.run()
    assert result.returncode != 0
    assert "YOLO" in result.stderr
    assert not _operations(deployment, "build")
    assert not _operations(deployment, "up")
    assert deployment.env_file.read_bytes() == before
    assert not (deployment.root / "releases" / "candidate").exists()
    assert not (deployment.root / ".deploy-lock").exists()


def test_base_face_assets_are_required_during_preflight(deployment: Deployment) -> None:
    asset = deployment.models / "insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
    asset.unlink()
    result = deployment.run("--check")
    assert result.returncode != 0
    assert "InsightFace" in result.stderr
    assert not deployment.root.exists()
    assert not _operations(deployment, "build")


def test_database_backup_failure_preserves_image_and_active_release(deployment: Deployment) -> None:
    previous = deployment.previous_release()
    before = deployment.env_file.read_bytes()
    result = deployment.run(flags={"MOCK_PREVIOUS_DATABASE": "1", "MOCK_BACKUP_FAIL": "1"})
    assert result.returncode != 0
    assert "database backup failed" in result.stderr
    assert deployment.env_file.read_bytes() == before
    assert (deployment.root / "active-release").read_text().strip() == str(previous)
    assert not _operations(deployment, "build")
    assert not _operations(deployment, "up")
    assert not (deployment.root / "releases" / "candidate").exists()
    backup = deployment.root / "backups" / "candidate"
    assert (backup / "config.env").read_bytes() == before
    assert (backup / "config.env").stat().st_mode & 0o777 == 0o600
    assert not (deployment.root / ".deploy-lock").exists()


def test_new_image_is_selected_verified_and_promoted_after_backup(deployment: Deployment) -> None:
    previous = deployment.previous_release()
    before = deployment.env_file.read_bytes()
    result = deployment.run(flags={"MOCK_PREVIOUS_DATABASE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    candidate = deployment.root / "releases" / "candidate"
    assert "SIGHTINDEX_IMAGE=sightindex:candidate\n" in deployment.env_file.read_text()
    assert (candidate / "SOURCE_REVISION").read_text().strip() == REVISION
    manifest = json.loads((candidate / "SNAPSHOT_MANIFEST.json").read_text())
    assert manifest["source_revision"] == REVISION
    assert (
        manifest["files_sha256"]["main.py"]
        == hashlib.sha256((deployment.source / "main.py").read_bytes()).hexdigest()
    )
    assert ".deployment-pending" not in manifest["files_sha256"]
    assert (candidate / "IMAGE_ID").read_text().strip() == CANDIDATE_ID
    assert (deployment.root / "active-release").read_text().strip() == str(candidate)
    assert (deployment.root / "active-stacks").read_text().strip() == "base"
    assert not (candidate / ".deployment-pending").exists()
    assert not (deployment.root / ".deploy-lock").exists()
    backup = deployment.root / "backups" / "candidate"
    assert (backup / "config.env").read_bytes() == before
    assert (backup / "previous-release").read_text().strip() == str(previous)
    assert (backup / "postgres.dump").read_text().startswith("PGDUMP-synthetic")
    commands = _docker_commands(deployment)
    backup_index = next(index for index, arguments in enumerate(commands) if "pg_dump" in arguments)
    build_index = next(index for index, arguments in enumerate(commands) if arguments[0] == "build")
    up_index = next(index for index, arguments in enumerate(commands) if "up" in arguments)
    verify_index = next(
        index
        for index, arguments in enumerate(commands)
        if "/opt/sightindex/deployment_verify.py" in arguments
    )
    assert backup_index < build_index < up_index < verify_index
    assert "API image verified" in result.stdout
    assert "--upload-smoke" in commands[verify_index]
    assert "--model-smoke" in commands[verify_index]
    assert commands[verify_index][-4:] == ["--stacks", "base", "--upload-smoke", "--model-smoke"]


@pytest.mark.parametrize("failure", ["MOCK_VERIFY_FAIL", "MOCK_IMAGE_MISMATCH", "MOCK_UP_FAIL"])
def test_candidate_failure_stays_pending_and_previous_active_marker_survives(
    deployment: Deployment, failure: str
) -> None:
    previous = deployment.previous_release()
    result = deployment.run(flags={"MOCK_PREVIOUS_DATABASE": "1", failure: "1"})
    assert result.returncode != 0
    assert (deployment.root / "active-release").read_text().strip() == str(previous)
    assert (deployment.root / "active-stacks").read_text() == "base\n"
    assert (deployment.root / "releases" / "candidate" / ".deployment-pending").is_file()
    assert "NOT accepted" in result.stderr
    assert "Services may be on the candidate version" in result.stderr
    assert not (deployment.root / ".deploy-lock").exists()
    if failure == "MOCK_IMAGE_MISMATCH":
        assert "API image mismatch" in result.stderr
        assert not _operations(deployment, "/opt/sightindex/deployment_verify.py")


def test_no_build_requires_existing_local_image_without_changing_env(
    deployment: Deployment,
) -> None:
    previous = deployment.previous_release()
    before = deployment.env_file.read_bytes()
    result = deployment.run("--no-build", flags={"MOCK_IMAGE_MISSING": "1"})
    assert result.returncode != 0
    assert "configured application image is not local" in result.stderr
    assert not _operations(deployment, "build")
    assert not _operations(deployment, "up")
    assert deployment.env_file.read_bytes() == before
    assert (deployment.root / "active-release").read_text().strip() == str(previous)


def test_no_build_accepts_only_verified_local_image(deployment: Deployment) -> None:
    before = deployment.env_file.read_bytes()
    result = deployment.run("--no-build")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not _operations(deployment, "build")
    assert deployment.env_file.read_bytes() == before
    assert (
        deployment.root / "releases" / "candidate" / "IMAGE_ID"
    ).read_text().strip() == PREVIOUS_ID
    assert _operations(deployment, "/opt/sightindex/deployment_verify.py")


def test_health_timeout_uses_bounded_wait_and_does_not_promote(deployment: Deployment) -> None:
    previous = deployment.previous_release()
    result = deployment.run(flags={"MOCK_UNHEALTHY": "1"})
    assert result.returncode != 0
    assert "selected services did not become healthy" in result.stderr
    # A slow first health poll can consume the one-second synthetic deadline.
    # That is a valid bounded timeout and need not execute another sleep.
    assert any(
        argument.startswith("{{.State.Status}}")
        for command in _docker_commands(deployment)
        for argument in command
    )
    assert not _operations(deployment, "/opt/sightindex/deployment_verify.py")
    assert (deployment.root / "active-release").read_text().strip() == str(previous)
    assert (deployment.root / "releases" / "candidate" / ".deployment-pending").is_file()
    assert not (deployment.root / ".deploy-lock").exists()


def test_dirty_source_requires_explicit_permission_before_mkdir(deployment: Deployment) -> None:
    before = deployment.env_file.read_bytes()
    result = deployment.run(flags={"MOCK_DIRTY": "1"})
    assert result.returncode != 0
    assert "--allow-dirty" in result.stderr
    assert not deployment.root.exists()
    assert deployment.env_file.read_bytes() == before
    assert not _operations(deployment, "build")


def test_allowed_dirty_source_is_labeled_on_image_and_release(deployment: Deployment) -> None:
    result = deployment.run("--allow-dirty", flags={"MOCK_DIRTY": "1"})
    assert result.returncode == 0, result.stderr
    revision = REVISION + "-dirty"
    candidate = deployment.root / "releases" / "candidate"
    assert (candidate / "SOURCE_REVISION").read_text().strip() == revision
    assert f"SOURCE_REVISION={revision}" in _operations(deployment, "build")[0]


@pytest.mark.parametrize("existing_previous", [False, True])
def test_existing_database_without_backup_capable_container_blocks_deployment(
    deployment: Deployment, existing_previous: bool
) -> None:
    before = deployment.env_file.read_bytes()
    if existing_previous:
        deployment.previous_release()
    result = deployment.run(flags={"MOCK_VOLUME_EXISTS": "1"})
    assert result.returncode != 0
    assert "verified external backup" in result.stderr
    assert deployment.env_file.read_bytes() == before
    assert not _operations(deployment, "build")
    assert not _operations(deployment, "up")


def test_explicit_external_backup_override_is_visible(deployment: Deployment) -> None:
    result = deployment.run("--skip-backup", flags={"MOCK_VOLUME_EXISTS": "1"})
    assert result.returncode == 0, result.stderr
    assert "database backup explicitly skipped" in result.stderr
    assert not _operations(deployment, "pg_dump")


def test_release_snapshot_excludes_private_env_databases_weights_and_media(
    deployment: Deployment,
) -> None:
    excluded = [
        ".env",
        ".env.private",
        "database.sqlite",
        "models/model.safetensors",
        "data/user-frame.jpg",
        "docs/operator-report.md",
        "app/.env.local",
        "app/legacy.db",
        "app/private.sqlite-wal",
        "app/private.db-shm",
        "app/private.db-journal",
        "app/data/user-frame.jpg",
        "frontend/node_modules/private-package/index.js",
        "frontend/dist/private-build.js",
        "deploy/containers/.env.operator",
        "deploy/agx/private-model.pt",
        "deploy/agx/private-model.pth",
        "deploy/agx/private-model.onnx",
        "deploy/agx/private-model.safetensors",
        "deploy/agx/private-model.safetensors.index.json",
        "deploy/agx/private-model.bin",
        "deploy/agx/weights/manifest.json",
        "deploy/agx/data/user-frame.jpg",
        "deploy/agx/cache/private.json",
    ]
    sentinel = b"fixture-private-content-must-not-enter-a-release"
    for relative in excluded:
        path = deployment.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(sentinel)
    result = deployment.run()
    assert result.returncode == 0, result.stdout + result.stderr
    candidate = deployment.root / "releases" / "candidate"
    assert (candidate / "deploy" / "containers" / ".env.example").is_file()
    assert (candidate / "main.py").is_file()
    assert all(not (candidate / relative).exists() for relative in excluded)
    assert not any(sentinel in path.read_bytes() for path in candidate.rglob("*") if path.is_file())


def test_source_symlink_cannot_put_external_private_content_into_release(
    deployment: Deployment, tmp_path: Path
) -> None:
    secret = tmp_path / "external-private-file"
    secret.write_text("synthetic external private content", encoding="utf-8")
    linked = deployment.source / "app" / "external.py"
    linked.symlink_to(secret)
    result = deployment.run()
    candidate = deployment.root / "releases" / "candidate"
    if result.returncode == 0:
        assert not (candidate / "app" / "external.py").exists()
        assert not any(path.is_symlink() for path in candidate.rglob("*"))
    else:
        assert not (deployment.root / "active-release").exists()
    assert secret.read_text() == "synthetic external private content"


def test_legitimate_frontend_static_assets_are_kept(deployment: Deployment) -> None:
    assets = ("frontend/src/assets/icon.png", "frontend/public/favicon.png")
    for relative in assets:
        path = deployment.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic frontend image")
    result = deployment.run()
    assert result.returncode == 0, result.stdout + result.stderr
    candidate = deployment.root / "releases" / "candidate"
    for relative in assets:
        assert (candidate / relative).read_bytes() == b"synthetic frontend image"


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_timeout_is_rejected_before_writes(deployment: Deployment, value: str) -> None:
    before = deployment.env_file.read_bytes()
    result = deployment.run(flags={"SIGHTINDEX_DEPLOY_TIMEOUT_SECONDS": value})
    assert result.returncode != 0
    assert "timeout must be a positive integer" in result.stderr
    assert not deployment.root.exists()
    assert deployment.env_file.read_bytes() == before
    assert not _operations(deployment, "build")
    assert not _operations(deployment, "up")


def test_root_dispatches_rtx_help_without_host_changes(deployment: Deployment) -> None:
    result = subprocess.run(
        ["/bin/bash", str(deployment.source / "deploy.sh"), "--target", "rtx5090", "--help"],
        env={"PATH": f"{deployment.tools}:/usr/bin:/bin"},
        cwd=deployment.source,
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert "install_or_update.sh" in result.stdout
    assert not deployment.root.exists()
    assert not deployment.commands()


def test_selected_stack_overlays_services_and_verifier_are_preserved(
    deployment: Deployment,
) -> None:
    assets = [
        "sapiensid_wb12m/model.pth",
        "sapiensid_wb12m/model.yaml",
        "yolov8n-pose.pt",
        "dfa_mobilenetv4_medium/mobilenetv4_Final.pth",
        "qwen/config.json",
        "qwen/tokenizer_config.json",
        "qwen/preprocessor_config.json",
        "qwen/tokenizer.json",
        "qwen/model.safetensors",
        "qwen/scripts/qwen3_vl_embedding.py",
    ]
    for relative in assets:
        path = deployment.models / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic model fixture")
    configuration = dict(
        line.split("=", 1) for line in deployment.env_file.read_text().splitlines()
    )
    configuration.update(
        REID_ENABLED="true",
        SEMANTIC_SEARCH_ENABLED="true",
        QWEN_EMBEDDING_API_KEY="fixture_private_embedding_key",
        QWEN_EMBEDDING_MODEL_DIR=str(deployment.models / "qwen"),
    )
    deployment.env_file.write_text(
        "".join(f"{key}={value}\n" for key, value in configuration.items()), encoding="utf-8"
    )
    stacks = "base reid embedding semantic"
    result = deployment.run("--stacks", stacks)
    assert result.returncode == 0, result.stdout + result.stderr
    up = _operations(deployment, "up")[0]
    assert "reid" in up and "embedding" in up
    assert any(argument.endswith("compose.embedding.yaml") for argument in up)
    assert any(argument.endswith("compose.semantic-search.yaml") for argument in up)
    acceptance = _operations(deployment, "/opt/sightindex/deployment_verify.py")[0]
    assert acceptance[acceptance.index("--stacks") + 1] == stacks
    assert (deployment.root / "active-stacks").read_text().strip() == stacks
    assert any(entry["tool"] == "nvidia-smi" for entry in deployment.commands())


@pytest.mark.parametrize("option", ["--source", "--root", "--stacks", "--release"])
def test_missing_option_value_is_rejected_before_writes(
    deployment: Deployment, option: str
) -> None:
    result = deployment.run(option)
    assert result.returncode != 0
    assert "missing value" in result.stderr
    assert not deployment.root.exists()


def test_deployment_lock_blocks_concurrent_changes(deployment: Deployment) -> None:
    lock = deployment.root / ".deploy-lock"
    lock.mkdir(parents=True)
    before = deployment.env_file.read_bytes()
    result = deployment.run()
    assert result.returncode != 0
    assert "deployment lock exists" in result.stderr
    assert lock.is_dir()
    assert deployment.env_file.read_bytes() == before
    assert not _operations(deployment, "build")
    assert not _operations(deployment, "up")


def test_only_new_writable_mount_directories_receive_owner_change(deployment: Deployment) -> None:
    existing = deployment.root / "media"
    existing.mkdir(parents=True)
    result = deployment.run(flags={"MOCK_UID": "501"})
    assert result.returncode == 0, result.stdout + result.stderr
    ownership = [entry["args"] for entry in deployment.commands() if entry["tool"] == "chown"]
    assert ownership == [
        ["1000:1000", str(deployment.root / "cache")],
        ["1000:1000", str(deployment.root / "cache" / "api")],
        ["1000:1000", str(deployment.env_file.parent / "assets" / "media")],
        ["1000:1000", str(deployment.env_file.parent / "assets" / "cache")],
        ["1000:1000", str(deployment.env_file.parent / "assets" / "cache" / "api")],
    ]
    assert all("-R" not in arguments for arguments in ownership)
    assert (deployment.root / "models").stat().st_mode & 0o777 == 0o755


def test_new_mount_permission_failure_stops_before_image_or_service_changes(
    deployment: Deployment,
) -> None:
    before = deployment.env_file.read_bytes()
    result = deployment.run(flags={"MOCK_UID": "501", "MOCK_CHOWN_FAIL": "1"})
    assert result.returncode != 0
    assert "container UID 1000" in result.stderr
    assert deployment.env_file.read_bytes() == before
    assert not _operations(deployment, "build")
    assert not _operations(deployment, "up")
    assert not (deployment.root / ".deploy-lock").exists()


def test_no_real_deployment_tool_can_be_reached(deployment: Deployment) -> None:
    """Every potentially external executable resolves into the temporary strict mock tree."""
    for name in ("docker", "git", "nvidia-smi", "fake-python", "id", "chown"):
        assert shutil.which(name, path=f"{deployment.tools}:/usr/bin:/bin") == str(
            deployment.tools / name
        )
    assert "ORCHESTRATION_ENV" not in os.environ


MODEL_SETUP_MOCK = """\
    elif args and args[0].endswith("/deploy/models/setup.py"):
        # Exercise only the real Bash orchestration, not downloads/model loading.
        fixture = Path(os.environ["ORCHESTRATION_ENV"]).parent
        assert Path(args[0]) == fixture / "source/deploy/models/setup.py"
        assert args[args.index("--source") + 1] == str(fixture / "source")
        assert args[args.index("--target") + 1] == "containers"
        assert args[args.index("--env-file") + 1] == os.environ["ORCHESTRATION_ENV"]
        manifest = Path(args[args.index("--manifest") + 1])
        assert manifest.is_relative_to(fixture) and manifest.is_file()
        if "--check" in args:
            preparation = {"--download", "--source-dir", "--acknowledge-model-terms"}
            assert not preparation.intersection(args)
        else:
            assert "--acknowledge-model-terms" in args
            assert ("--download" in args) != ("--source-dir" in args)
            if "--source-dir" in args:
                assert Path(args[args.index("--source-dir") + 1]) == fixture / "bundle"
        if flag("MOCK_MODEL_SETUP_FAIL"):
            print("synthetic model integrity failure", file=sys.stderr)
            raise SystemExit(61)
        print(json.dumps({"mode": "check" if "--check" in args else "prepare", "ok": True}))
"""


@pytest.fixture
def model_deployment(deployment: Deployment) -> Deployment:
    """Add real model tools but stub only their subprocess execution boundary."""
    model_tools = deployment.source / "deploy" / "models"
    model_tools.mkdir()
    for name in ("manage.sh", "setup.py", "assets.py"):
        shutil.copyfile(REPOSITORY / "deploy" / "models" / name, model_tools / name)
    containers = deployment.source / "deploy" / "containers"
    for name in ("Dockerfile.embedding", "embedding_app.py", "download_embedding_model.py"):
        shutil.copyfile(REPOSITORY / "deploy" / "containers" / name, containers / name)
    boundary = '    elif args and args[0].endswith("preflight.py"):\n'
    assert MOCK_TOOL.count(boundary) == 1
    executable = deployment.tools / "fake-python"
    executable.write_text(
        f"#!{sys.executable}\n" + MOCK_TOOL.replace(boundary, MODEL_SETUP_MOCK + boundary),
        encoding="utf-8",
    )
    return deployment


@pytest.fixture
def reviewed_model_manifest(model_deployment: Deployment) -> Path:
    """Build a synthetic six-role base bundle; no downloaded weights are used."""
    bundle = model_deployment.env_file.parent / "bundle"
    entries: list[dict[str, Any]] = []
    files = {"yolo.person": model_deployment.models / "yolo11n.pt"}
    face = model_deployment.models / "insightface" / "models" / "buffalo_l"
    for name in ("det_10g", "w600k_r50", "1k3d68", "2d106det", "genderage"):
        target = face / f"{name}.onnx"
        if not target.exists():
            target.write_bytes(b"synthetic face fixture; not usable weights")
        files[f"face.{name}"] = target
    for identifier, target in files.items():
        relative = f"base/{target.name}"
        source = bundle / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(target.read_bytes())
        entries.append(
            {
                "id": identifier,
                "path": relative,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "size_bytes": source.stat().st_size,
                "model": "yolo11n" if identifier == "yolo.person" else "buffalo_l",
                "revision": "synthetic-reviewed-test-revision",
                "license": "synthetic-test-assets-only",
                "terms_url": "https://publisher.example.invalid/test-terms",
                "source_url": "https://publisher.example.invalid/test-assets/" + target.name,
            }
        )
    manifest = bundle / "models.lock.json"
    manifest.write_text(json.dumps({"version": 1, "artifacts": entries}), encoding="utf-8")
    return manifest


def _model_setup_commands(deployment: Deployment) -> list[list[str]]:
    """Select recorded model-tool invocations without reading a real environment."""
    return [
        entry["args"]
        for entry in deployment.commands()
        if entry["tool"] == "fake-python"
        and entry["args"]
        and entry["args"][0].endswith("/deploy/models/setup.py")
    ]


def _seed_embedding_profile(deployment: Deployment) -> None:
    """Prepare only synthetic Qwen bytes and private fixture configuration."""
    for relative in (
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "model.safetensors",
        "scripts/qwen3_vl_embedding.py",
    ):
        target = deployment.models / "qwen" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"synthetic model fixture; not usable weights")
    values = dict(line.split("=", 1) for line in deployment.env_file.read_text().splitlines())
    values.update(
        QWEN_EMBEDDING_MODEL_DIR=str(deployment.models / "qwen"),
        QWEN_EMBEDDING_API_KEY="fixture_private_embedding_key",
        QWEN_EMBEDDING_IMAGE="sightindex-embedding:reviewed-fixture",
    )
    deployment.env_file.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8"
    )


@pytest.mark.parametrize(
    "arguments,error",
    [
        (("--prepare-models",), "requires --model-manifest"),
        (
            ("--prepare-models", "--model-manifest", "MANIFEST"),
            "requires --acknowledge-model-terms",
        ),
        (
            ("--prepare-models", "--model-manifest", "MANIFEST", "--acknowledge-model-terms"),
            "select --model-source or --download-models",
        ),
        (("--model-source", "BUNDLE"), "options require --prepare-models"),
        (("--acknowledge-model-terms",), "options require --prepare-models"),
        (("--download-models",), "requires --model-manifest"),
        (
            (
                "--prepare-models",
                "--model-manifest",
                "MANIFEST",
                "--model-source",
                "BUNDLE",
                "--download-models",
                "--acknowledge-model-terms",
            ),
            "choose offline model source or download, not both",
        ),
    ],
)
def test_model_preparation_flags_fail_before_any_deployment_write(
    model_deployment: Deployment,
    reviewed_model_manifest: Path,
    arguments: tuple[str, ...],
    error: str,
) -> None:
    """Model authority/source omissions are rejected before mkdir or tools."""
    replacements = {
        "MANIFEST": str(reviewed_model_manifest),
        "BUNDLE": str(reviewed_model_manifest.parent),
    }
    before = model_deployment.env_file.read_bytes()
    model_before = {
        path: path.read_bytes() for path in model_deployment.models.rglob("*") if path.is_file()
    }
    result = model_deployment.run(*(replacements.get(value, value) for value in arguments))
    assert result.returncode != 0
    assert error in result.stderr
    assert not model_deployment.root.exists()
    assert model_deployment.env_file.read_bytes() == before
    assert all(path.read_bytes() == content for path, content in model_before.items())
    assert not model_deployment.commands()


def test_root_model_only_help_creates_nothing(model_deployment: Deployment) -> None:
    """The actual public model-only adapter exits before Python or Docker."""
    before = model_deployment.env_file.read_bytes()
    result = subprocess.run(
        ["/bin/bash", str(model_deployment.source / "deploy.sh"), "--models-only", "--help"],
        env={"PATH": f"{model_deployment.tools}:/usr/bin:/bin"},
        cwd=model_deployment.source,
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert "--models-only" in result.stdout and "--download-models" in result.stdout
    assert "not permission for commercial use" in result.stdout
    assert not model_deployment.root.exists()
    assert model_deployment.env_file.read_bytes() == before
    assert not model_deployment.commands()


@pytest.mark.parametrize("preparation_requested", [False, True])
def test_manifest_check_is_readonly_even_with_preparation_flags(
    model_deployment: Deployment,
    reviewed_model_manifest: Path,
    preparation_requested: bool,
) -> None:
    """Normal deployment check strips preparation authority and calls check only."""
    before = model_deployment.env_file.read_bytes()
    manifest_before = reviewed_model_manifest.read_bytes()
    arguments = ["--check", "--model-manifest", str(reviewed_model_manifest)]
    if preparation_requested:
        arguments.extend(
            [
                "--prepare-models",
                "--model-source",
                str(reviewed_model_manifest.parent),
                "--acknowledge-model-terms",
            ]
        )
    result = model_deployment.run(*arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not model_deployment.root.exists()
    assert model_deployment.env_file.read_bytes() == before
    assert reviewed_model_manifest.read_bytes() == manifest_before
    calls = _model_setup_commands(model_deployment)
    assert len(calls) == 1 and calls[0][-1] == "--check"
    assert not {"--source-dir", "--download", "--acknowledge-model-terms"}.intersection(calls[0])
    assert _docker_commands(model_deployment) == [
        ["compose", "version"],
        ["info", "--format", "{{.OSType}}"],
    ]
    assert not any(entry["tool"] == "git" for entry in model_deployment.commands())


def test_failed_manifest_check_does_not_reach_preflight_tools(
    model_deployment: Deployment, reviewed_model_manifest: Path
) -> None:
    """An integrity failure stops a read-only deployment before any Docker call."""
    result = model_deployment.run(
        "--check",
        "--model-manifest",
        str(reviewed_model_manifest),
        flags={"MOCK_MODEL_SETUP_FAIL": "1"},
    )
    assert result.returncode != 0
    assert "synthetic model integrity failure" in result.stderr
    assert not model_deployment.root.exists()
    assert not _docker_commands(model_deployment)


def test_failed_model_preparation_blocks_build_and_preserves_previous_active(
    model_deployment: Deployment, reviewed_model_manifest: Path
) -> None:
    """No image or service mutation follows a model-prepare failure."""
    previous = model_deployment.previous_release()
    before = model_deployment.env_file.read_bytes()
    result = model_deployment.run(
        "--prepare-models",
        "--model-manifest",
        str(reviewed_model_manifest),
        "--model-source",
        str(reviewed_model_manifest.parent),
        "--acknowledge-model-terms",
        flags={"MOCK_MODEL_SETUP_FAIL": "1"},
    )
    assert result.returncode != 0
    assert "synthetic model integrity failure" in result.stderr
    assert not _operations(model_deployment, "build")
    assert not _operations(model_deployment, "up")
    assert not _operations(model_deployment, "pg_dump")
    assert not (model_deployment.root / "releases" / "candidate").exists()
    assert model_deployment.env_file.read_bytes() == before
    assert (model_deployment.root / "active-release").read_text().strip() == str(previous)
    assert not (model_deployment.root / ".deploy-lock").exists()


@pytest.mark.parametrize("download", [False, True])
def test_model_preparation_precedes_preflight_and_image_build(
    model_deployment: Deployment, reviewed_model_manifest: Path, download: bool
) -> None:
    """Both explicit sources are forwarded before deploy; HTTP remains stubbed."""
    source = (
        ["--download-models"]
        if download
        else ["--model-source", str(reviewed_model_manifest.parent)]
    )
    result = model_deployment.run(
        "--prepare-models",
        "--model-manifest",
        str(reviewed_model_manifest),
        "--acknowledge-model-terms",
        *source,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = _model_setup_commands(model_deployment)
    assert len(calls) == 1
    assert "--check" not in calls[0]
    assert "--acknowledge-model-terms" in calls[0]
    assert ("--download" in calls[0]) == download
    assert ("--source-dir" in calls[0]) != download
    commands = model_deployment.commands()
    setup_index = next(index for index, entry in enumerate(commands) if entry["args"] == calls[0])
    preflight_index = next(
        index
        for index, entry in enumerate(commands)
        if entry["tool"] == "fake-python"
        and entry["args"][0].endswith("preflight.py")
        and "--check-tools" in entry["args"]
    )
    build_index = next(
        index
        for index, entry in enumerate(commands)
        if entry["tool"] == "docker" and entry["args"][0] == "build"
    )
    assert setup_index < preflight_index < build_index


def test_manifest_without_preparation_only_checks_existing_models(
    model_deployment: Deployment, reviewed_model_manifest: Path
) -> None:
    """Supplying a lockfile never implicitly enables copying or downloading."""
    result = model_deployment.run("--model-manifest", str(reviewed_model_manifest))
    assert result.returncode == 0, result.stdout + result.stderr
    calls = _model_setup_commands(model_deployment)
    assert len(calls) == 1 and calls[0][-1] == "--check"
    assert not {"--source-dir", "--download", "--acknowledge-model-terms"}.intersection(calls[0])
    assert _operations(model_deployment, "build") and _operations(model_deployment, "up")


@pytest.mark.parametrize("stacks", ["base", "base reid"])
def test_model_service_build_requires_embedding_before_writes(
    model_deployment: Deployment, stacks: str
) -> None:
    """No model image is built when its capability was not selected."""
    before = model_deployment.env_file.read_bytes()
    result = model_deployment.run("--stacks", stacks, "--build-model-services")
    assert result.returncode != 0
    assert "requires embedding stack" in result.stderr
    assert not model_deployment.root.exists()
    assert model_deployment.env_file.read_bytes() == before
    assert not model_deployment.commands()


@pytest.mark.parametrize("build_application", [False, True])
def test_embedding_image_build_uses_selected_application_image(
    model_deployment: Deployment, build_application: bool
) -> None:
    """The model image inherits this release's app image, not a stale source tag."""
    _seed_embedding_profile(model_deployment)
    arguments = ["--stacks", "base embedding", "--build-model-services"]
    if not build_application:
        arguments.append("--no-build")
    result = model_deployment.run(*arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    builds = _operations(model_deployment, "build")
    assert len(builds) == (2 if build_application else 1)
    embedding = next(
        arguments
        for arguments in builds
        if any(value.endswith("Dockerfile.embedding") for value in arguments)
    )
    expected_image = "sightindex:candidate" if build_application else "sightindex:old"
    assert (
        embedding[embedding.index("--build-arg") + 1] == f"SIGHTINDEX_BASE_IMAGE={expected_image}"
    )
    assert embedding[embedding.index("-t") + 1] == "sightindex-embedding:reviewed-fixture"
    candidate = model_deployment.root / "releases" / "candidate"
    assert embedding[-1] == str(candidate)
    assert (candidate / "deploy" / "containers" / "Dockerfile.embedding").is_file()
    if build_application:
        assert "SOURCE_REVISION=" + REVISION in builds[0]
        assert builds.index(embedding) == 1
    assert _operations(model_deployment, "up")
    assert _operations(model_deployment, "/opt/sightindex/deployment_verify.py")
    image_checks = [
        command
        for command in _docker_commands(model_deployment)
        if command[:3] == ["inspect", "--format", "{{.Image}}"]
    ]
    assert {command[-1] for command in image_checks} == {"aa" * 6, "ff" * 6}


def test_snapshot_keeps_model_tools_but_not_model_binary_assets(
    model_deployment: Deployment,
) -> None:
    """The deploy/models code exception cannot admit weights or runtime data."""
    excluded = [
        "deploy/models/model.pt",
        "deploy/models/model.pt.partial",
        "deploy/models/model.pth",
        "deploy/models/model.pth.partial",
        "deploy/models/model.onnx",
        "deploy/models/model.onnx.partial",
        "deploy/models/model.safetensors",
        "deploy/models/model.safetensors.partial",
        "deploy/models/model.safetensors.index.json",
        "deploy/models/model.bin",
        "deploy/models/model.bin.partial",
        "deploy/models/.sightindex-model-assets.lock",
        "deploy/agx/.sightindex-model-assets.lock",
        "deploy/models/.sightindex-model-assets-quarantine/interrupted/legacy.lock",
        "deploy/agx/.sightindex-model-assets-quarantine/interrupted/recovery.json",
        "deploy/agx/reid_service/.sightindex-model-assets-quarantine/interrupted/lock-v2",
        "app/.sightindex-model-assets-quarantine/interrupted/no-extension",
        "deploy/agx/reid_service/sapiensid/tasks/sapiensID/src/aligners/"
        "keypoint_predictor/pretrained_models/aligners/dfa_mobilenetv4_medium/"
        "mobilenetv4_Final.pth.partial",
        "deploy/models/.env.private",
        "deploy/models/private.sqlite-wal",
        "deploy/models/weights/manifest.json",
        "deploy/models/data/private-frame.jpg",
        "models/model.safetensors",
    ]
    private = b"fixture-private-model-content-not-allowed-in-source-snapshot"
    for relative in excluded:
        target = model_deployment.source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(private)
    result = model_deployment.run()
    assert result.returncode == 0, result.stdout + result.stderr
    candidate = model_deployment.root / "releases" / "candidate"
    manifest = json.loads((candidate / "SNAPSHOT_MANIFEST.json").read_text())
    for name in ("manage.sh", "setup.py", "assets.py"):
        relative = f"deploy/models/{name}"
        expected = (model_deployment.source / relative).read_bytes()
        assert (candidate / relative).read_bytes() == expected
        assert manifest["files_sha256"][relative] == hashlib.sha256(expected).hexdigest()
    assert all(not (candidate / relative).exists() for relative in excluded)
    assert not any(
        private in target.read_bytes() for target in candidate.rglob("*") if target.is_file()
    )
