"""Legacy source bundles include code, never mutable deployment runtime state."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

from deploy.rtx5090 import legacy_bundle as bundle


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    for name, content in {
        "main.py": "from app import api\n",
        "app/__init__.py": "",
        "app/models/media.py": "class Media: pass\n",
        "app/db/session.py": "def create_session(): pass\n",
        "app/services/video_playback.py": "READY = True\n",
        "frontend/dist/index.html": '<script src="/assets/index-abc.js"></script>',
        "frontend/dist/assets/index-abc.js": "export const ready = true;",
        "frontend/src/main.ts": "export const ready = true;",
        "frontend/package.json": '{"name":"fixture"}',
        "frontend/package-lock.json": "{}",
        "deploy/rtx5090/legacy_acceptance.py": "def main(): return 0\n",
        "deploy/models/setup.py": "def main(): return 0\n",
        "deploy/containers/Dockerfile": "FROM fixture\n",
        "docs/model-deployment.zh-CN.md": "模型说明\n",
        "README.zh-CN.md": "使用说明\n",
        "requirements.rtx5090.txt": "fixture==1.0\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def _write(source: Path, name: str, content: bytes = b"PRIVATE") -> Path:
    path = source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _build(source: Path, tmp_path: Path) -> tuple[Path, str, dict[str, Any]]:
    output = tmp_path / "release.tar.gz"
    digest = bundle.build_bundle(source, output, "36a4adc-dirty-fixture")
    return output, digest, bundle.verify_bundle(output, expected_manifest_sha256=digest)


def _rewrite_archive(output: Path, changes: dict[str, tuple[bytes, int]]) -> None:
    members: list[tuple[str, bytes, int]] = []
    with tarfile.open(output, "r:gz") as archive:
        for member in archive.getmembers():
            handle = archive.extractfile(member)
            assert handle is not None
            members.append((member.name, handle.read(), member.mode))
    names = {name for name, _, _ in members}
    with tarfile.open(output, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        for name, content, mode in members:
            content, mode = changes.get(name, (content, mode))
            bundle._tar_member(archive, name, content, mode)
        for name in changes.keys() - names:
            content, mode = changes[name]
            bundle._tar_member(archive, name, content, mode)


def test_code_static_build_and_manifest_identity(source: Path, tmp_path: Path) -> None:
    executable = _write(source, "deploy/rtx5090/verify.sh", b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o751)
    _write(source, "frontend/dist/assets/icon.svg", b'<svg xmlns="http://www.w3.org/2000/svg"/>')
    output, digest, manifest = _build(source, tmp_path)
    assert output.stat().st_mode & 0o777 == 0o600
    assert manifest["revision"] == "36a4adc-dirty-fixture"
    assert manifest["dirty"] is True
    assert ".env" in manifest["protected_paths"]
    entries = {entry["path"]: entry for entry in manifest["entries"]}
    assert "app/services/video_playback.py" in entries
    assert "app/models/media.py" in entries
    assert "app/db/session.py" in entries
    assert "deploy/models/setup.py" in entries
    assert "deploy/rtx5090/legacy_acceptance.py" in entries
    assert "frontend/dist/assets/icon.svg" in entries
    assert entries["deploy/rtx5090/verify.sh"]["mode"] == "0755"
    assert entries["main.py"]["mode"] == "0644"
    with tarfile.open(output, "r:gz") as archive:
        handle = archive.extractfile(bundle.MANIFEST_NAME)
        assert handle is not None
        assert hashlib.sha256(handle.read()).hexdigest() == digest
        assert all(member.isfile() and member.uid == 0 and member.gid == 0 for member in archive)


def test_only_reviewed_operator_scripts_are_released(source: Path, tmp_path: Path) -> None:
    for name in bundle.OPERATOR_FILES:
        _write(source, name, b"# synthetic reviewed operator code\n")
    for name in ("docs/reid-walkthrough-calibration.md", "docs/reid-walkthrough-template.csv"):
        _write(source, name, b"synthetic walkthrough documentation\n")
    _write(source, "scripts/private-operation.py")
    _write(source, "scripts/reid-feedback.csv")
    _write(source, "scripts/.env")
    _, _, manifest = _build(source, tmp_path)
    names = {entry["path"] for entry in manifest["entries"]}
    assert bundle.OPERATOR_FILES <= names
    assert "docs/reid-walkthrough-calibration.md" in names
    assert "docs/reid-walkthrough-template.csv" in names
    assert "scripts/private-operation.py" not in names
    assert "scripts/reid-feedback.csv" not in names
    assert "scripts/.env" not in names


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.production",
        "sightindex.db",
        "sightindex.db-wal",
        "sightindex.db-shm",
        "data/crop.png",
        "models/model.pth",
        "backups/secret.py",
        "logs/log.py",
        ".venv/settings.py",
        "app/__pycache__/code.py",
        "app/cache/secret.py",
        "app/uploads/image.py",
        "app/secret.pem",
        "app/secret.key",
        "app/model.pt",
        "app/data/models/media.py",
        "app/models/model.pt",
        "app/model.safetensors",
        "app/secret.env",
        "app/.env",
        "deploy/models/model.pth",
        "deploy/models/model.onnx",
        "deploy/models/file.bin",
        "deploy/models/model.partial",
        "deploy/models/.sightindex-model-assets.lock",
        "deploy/models/.sightindex-model-assets-quarantine/secret.py",
        "deploy/rtx5090/backups/private.py",
        "deploy/rtx5090/site.env",
        "deploy/agx/vendor/code.py",
        "frontend/node_modules/private.js",
        "frontend/dist/assets/video.mp4",
        "frontend/dist/media/image.png",
        "frontend/src/user-images/person.png",
        "frontend/src/photo.png",
        "frontend/public/photo.png",
        "docs/overall-effect-assessment-20260908.md",
        "docs/overall-effect-assessment-20260908.docx",
        "docs/private.md",
        "tests/private.py",
        "private.py",
    ],
)
def test_runtime_and_reports_are_never_included(source: Path, tmp_path: Path, name: str) -> None:
    _write(source, name)
    _, _, manifest = _build(source, tmp_path)
    assert name not in {entry["path"] for entry in manifest["entries"]}


def test_only_tracked_templates_and_native_assets_are_included(
    source: Path, tmp_path: Path
) -> None:
    for name in [
        ".env.example",
        "deploy/containers/.env.example",
        "deploy/rtx5090/sightindex.env.example",
        "frontend/public/icon.png",
        "frontend/src/assets/icon.svg",
    ]:
        _write(source, name, b"fixture")
    _write(source, "deploy/models/private.env.example", b"must-not-include")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "add",
            ".env.example",
            "deploy/containers/.env.example",
            "deploy/rtx5090/sightindex.env.example",
            "frontend/public/icon.png",
            "frontend/src/assets/icon.svg",
        ],
        check=True,
    )
    _, _, manifest = _build(source, tmp_path)
    names = {entry["path"] for entry in manifest["entries"]}
    assert {
        ".env.example",
        "deploy/containers/.env.example",
        "deploy/rtx5090/sightindex.env.example",
        "frontend/public/icon.png",
        "frontend/src/assets/icon.svg",
    } <= names
    assert "deploy/models/private.env.example" not in names


@pytest.mark.parametrize(
    "name",
    [
        "app/link.py",
        "app/cache",
        "frontend/dist/assets/link.js",
        "deploy/models/link.py",
        "README.md",
    ],
)
def test_symlinks_fail_closed(source: Path, tmp_path: Path, name: str) -> None:
    path = source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(tmp_path / "nonexistent-private-target")
    output = tmp_path / "release.tar.gz"
    with pytest.raises((bundle.BundleError, OSError)):
        bundle.build_bundle(source, output, "36a4adc-dirty-fixture")
    assert not output.exists()


@pytest.mark.parametrize(
    "name",
    ["app", "app/models/media.py", "app/db/session.py", "main.py", "frontend/dist/index.html"],
)
def test_incomplete_source_is_not_published(source: Path, tmp_path: Path, name: str) -> None:
    path = source / name
    if path.is_dir():
        path.rename(source / "excluded-app")
    else:
        path.unlink()
    with pytest.raises(bundle.BundleError, match="requires"):
        bundle.build_bundle(source, tmp_path / "release.tar.gz", "36a4adc-dirty-fixture")


@pytest.mark.parametrize(
    "label",
    [
        "36a4adc",
        "36a4adc-clean",
        "release",
        "36a4adc-dirty-../../escape",
        "36a4adc-dirty-",
        "36a4adc-dirty-release\n",
        "36a4adc-dirty-SECRET=private",
    ],
)
def test_revision_cannot_misrepresent_clean_commit_or_escape(
    source: Path, tmp_path: Path, label: str
) -> None:
    with pytest.raises(bundle.BundleError, match="revision"):
        bundle.build_bundle(source, tmp_path / "release.tar.gz", label)


def test_output_does_not_overwrite_file_or_dangling_link(source: Path, tmp_path: Path) -> None:
    output = tmp_path / "release.tar.gz"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        bundle.build_bundle(source, output, "36a4adc-dirty-fixture")
    assert output.read_bytes() == b"keep"
    output.unlink()
    output.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        bundle.build_bundle(source, output, "36a4adc-dirty-fixture")
    assert output.is_symlink()


def test_output_is_outside_source_and_parent_must_exist(source: Path, tmp_path: Path) -> None:
    with pytest.raises(bundle.BundleError, match="outside"):
        bundle.build_bundle(source, source / "bundle.tar.gz", "36a4adc-dirty-fixture")
    with pytest.raises(bundle.BundleError, match="parent"):
        bundle.build_bundle(source, tmp_path / "missing" / "bundle.tar.gz", "36a4adc-dirty-fixture")


def test_atomic_publish_race_preserves_competing_output(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "release.tar.gz"
    original_link = os.link

    def competing_link(src: Path, dst: Path) -> None:
        dst.write_bytes(b"competing-release")
        original_link(src, dst)

    monkeypatch.setattr(bundle.os, "link", competing_link)
    with pytest.raises(FileExistsError):
        bundle.build_bundle(source, output, "36a4adc-dirty-fixture")
    assert output.read_bytes() == b"competing-release"
    assert not list(tmp_path.glob(".sightindex-bundle-*"))


@pytest.mark.parametrize("change", ["bytes", "mode", "extra"])
def test_verification_checks_exact_bytes_modes_and_inventory(
    source: Path, tmp_path: Path, change: str
) -> None:
    output, digest, _ = _build(source, tmp_path)
    changes = {
        "bytes": {"main.py": (b"different", 0o644)},
        "mode": {"main.py": ((source / "main.py").read_bytes(), 0o755)},
        "extra": {"extra.py": (b"private", 0o644)},
    }
    _rewrite_archive(output, changes[change])
    with pytest.raises(bundle.BundleError, match="identity|inventory"):
        bundle.verify_bundle(output, expected_manifest_sha256=digest)


def test_verification_checks_manifest_sha(source: Path, tmp_path: Path) -> None:
    output, _, _ = _build(source, tmp_path)
    with pytest.raises(bundle.BundleError, match="manifest SHA256"):
        bundle.verify_bundle(output, expected_manifest_sha256="0" * 64)


@pytest.mark.parametrize(
    "name", ["../escape.py", "/absolute.py", "app/../escape.py", "app//escape.py", "app\\escape.py"]
)
def test_verification_rejects_unsafe_archive_paths(source: Path, tmp_path: Path, name: str) -> None:
    output, _, _ = _build(source, tmp_path)
    _rewrite_archive(output, {name: (b"escape", 0o644)})
    with pytest.raises(bundle.BundleError, match="unsafe archive path"):
        bundle.verify_bundle(output)


def test_verification_rejects_link_members(source: Path, tmp_path: Path) -> None:
    output, _, _ = _build(source, tmp_path)
    with tarfile.open(output, "w:gz") as archive:
        member = tarfile.TarInfo("main.py")
        member.type = tarfile.SYMTYPE
        member.linkname = "/private/secret"
        archive.addfile(member)
    with pytest.raises(bundle.BundleError, match="regular files"):
        bundle.verify_bundle(output)


def test_manifest_cannot_authorize_private_runtime_path(source: Path, tmp_path: Path) -> None:
    output, _, manifest = _build(source, tmp_path)
    content = b"private"
    manifest["entries"].append(
        {
            "path": ".env",
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "mode": "0644",
        }
    )
    _rewrite_archive(
        output,
        {bundle.MANIFEST_NAME: (json.dumps(manifest).encode(), 0o644), ".env": (content, 0o644)},
    )
    with pytest.raises(bundle.BundleError, match="disallowed path"):
        bundle.verify_bundle(output)


def test_cli_build_and_read_only_verify(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "release.tar.gz"
    assert (
        bundle.main(
            [
                "--source",
                str(source),
                "--output",
                str(output),
                "--revision",
                "36a4adc-dirty-fixture",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    before = output.read_bytes()
    assert (
        bundle.main(["--verify", str(output), "--manifest-sha256", report["manifest_sha256"]]) == 0
    )
    assert json.loads(capsys.readouterr().out)["verified"] is True
    assert output.read_bytes() == before


def test_cli_does_not_expose_source_exception_or_content(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        bundle.main(
            [
                "--source",
                str(source),
                "--output",
                str(tmp_path / "missing" / "private-secret.tar.gz"),
                "--revision",
                "36a4adc-dirty-fixture",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out) == {"error": "unsafe_or_invalid_source_bundle"}


def test_same_snapshot_is_deterministic(source: Path, tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    assert bundle.build_bundle(source, first, "36a4adc-dirty-fixture") == bundle.build_bundle(
        source, second, "36a4adc-dirty-fixture"
    )
    assert first.read_bytes() == second.read_bytes()


def test_file_and_total_size_limits(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bundle, "MAX_FILE_BYTES", 2)
    with pytest.raises(bundle.BundleError, match="oversized"):
        bundle.build_bundle(source, tmp_path / "release.tar.gz", "36a4adc-dirty-fixture")
    monkeypatch.setattr(bundle, "MAX_FILE_BYTES", 1024 * 1024)
    monkeypatch.setattr(bundle, "MAX_BUNDLE_BYTES", 2)
    with pytest.raises(bundle.BundleError, match="safety limits"):
        bundle.build_bundle(source, tmp_path / "release.tar.gz", "36a4adc-dirty-fixture")


@pytest.mark.parametrize("field,value", [("mode", []), ("bytes", True), ("sha256", "bad")])
def test_malformed_manifest_values_fail_safely(
    source: Path, tmp_path: Path, field: str, value: Any
) -> None:
    output, _, manifest = _build(source, tmp_path)
    manifest["entries"][0][field] = value
    _rewrite_archive(output, {bundle.MANIFEST_NAME: (json.dumps(manifest).encode(), 0o644)})
    with pytest.raises(bundle.BundleError, match="invalid manifest identity"):
        bundle.verify_bundle(output)


def test_manifest_cannot_be_executable(source: Path, tmp_path: Path) -> None:
    output, _, manifest = _build(source, tmp_path)
    _rewrite_archive(output, {bundle.MANIFEST_NAME: (json.dumps(manifest).encode(), 0o755)})
    with pytest.raises(bundle.BundleError, match="permissions"):
        bundle.verify_bundle(output)


def test_real_source_snapshot_contains_all_backend_python_and_required_orm() -> None:
    root = Path(__file__).resolve().parents[1]
    snapshots = bundle._snapshot(root)
    names = {item.path for item in snapshots}
    expected = {
        path.relative_to(root).as_posix()
        for path in (root / "app").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    assert expected <= names, sorted(expected - names)
    assert {"app/models/media.py", "app/db/session.py"} <= names
    assert not any(name.startswith("deploy/agx/") for name in names)
    assert not any("overall-effect-assessment" in name for name in names)
