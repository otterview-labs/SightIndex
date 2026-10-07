"""Build a code-only, verifiable bundle for a legacy in-place deployment.

This helper never reads application settings, contacts a server, deletes target
files, or includes runtime state. A dirty release label is intentional: a local
working-tree snapshot must not be presented as the contents of a clean commit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_NAME = "SOURCE_MANIFEST.json"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_FILE_COUNT = 10_000
REVISION_PATTERN = re.compile(r"[a-f0-9]{7,40}-dirty-[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
ROOT_FILES = frozenset(
    {
        "main.py",
        "deploy.sh",
        ".dockerignore",
        ".env.example",
        "pyproject.toml",
        "uv.lock",
        "THIRD_PARTY_NOTICES.md",
        "LICENSE",
        "LICENSE.md",
        "LICENSE.txt",
    }
)
DOCUMENTS = frozenset(
    {
        "docs/deployment.md",
        "docs/deployment.zh-CN.md",
        "docs/model-deployment.zh-CN.md",
        "docs/one-click-deployment.zh-CN.md",
        "docs/reid-walkthrough-calibration.md",
        "docs/reid-walkthrough-template.csv",
    }
)
OPERATOR_FILES = frozenset(
    {
        "scripts/evaluate_reid_walkthrough.py",
        "scripts/cleanup_media_before.py",
        "scripts/cleanup_non_crossing_stream_frames.py",
    }
)
FRONTEND_FILES = frozenset(
    {
        "frontend/package.json",
        "frontend/package-lock.json",
        "frontend/index.html",
        "frontend/vite.config.ts",
        "frontend/tsconfig.json",
        "frontend/tsconfig.app.json",
        "frontend/tsconfig.node.json",
    }
)
TREES = (
    "app",
    "frontend/src",
    "frontend/scripts",
    "frontend/public",
    "frontend/dist",
    "deploy/containers",
    "deploy/models",
    "deploy/rtx5090",
)
CODE_EXTENSIONS = frozenset({".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json"})
FRONTEND_EXTENSIONS = frozenset({".vue", ".ts", ".js", ".mjs", ".cjs", ".css", ".json"})
STATIC_EXTENSIONS = frozenset(
    {
        ".svg",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".avif",
        ".gif",
        ".ico",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
    }
)
DIST_EXTENSIONS = STATIC_EXTENSIONS | frozenset({".html", ".js", ".css", ".map"})
BLOCKED_COMPONENTS = frozenset(
    {
        ".git",
        ".venv",
        ".venv-rtx5090",
        "venv",
        "node_modules",
        "data",
        "models",
        "cache",
        ".cache",
        "backups",
        "logs",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".sightindex-model-assets-quarantine",
        "quarantine",
        "uploads",
        "user-images",
        "media",
    }
)
BLOCKED_SUFFIXES = frozenset(
    {
        ".db",
        ".sqlite",
        ".sqlite3",
        ".db-wal",
        ".db-shm",
        ".sqlite-wal",
        ".sqlite-shm",
        ".sqlite3-wal",
        ".sqlite3-shm",
        ".pt",
        ".pth",
        ".onnx",
        ".safetensors",
        ".bin",
        ".partial",
        ".lock",
        ".pyc",
        ".pyo",
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".webm",
        ".mpeg",
        ".mpg",
        ".m4v",
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".crt",
        ".docx",
        ".pdf",
    }
)
PROTECTED_PATHS = (
    ".env",
    "sightindex.db",
    "data/",
    "models/",
    ".cache/",
    ".venv/",
    "backups/",
    "deploy/agx/reid_service/",
)


class BundleError(ValueError):
    """A bundle cannot be safely created or verified."""


@dataclass(frozen=True)
class _Snapshot:
    path: str
    content: bytes
    mode: int

    def entry(self) -> dict[str, str | int]:
        """Return the JSON-safe identity of this immutable file snapshot."""
        return {
            "path": self.path,
            "sha256": hashlib.sha256(self.content).hexdigest(),
            "bytes": len(self.content),
            "mode": f"{self.mode:04o}",
        }


def _safe_path(value: str) -> PurePosixPath:
    """Reject archive names that are not canonical relative POSIX paths."""
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or any(ord(char) < 32 for char in value)
        or path.is_absolute()
        or str(path) != value
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise BundleError("unsafe archive path")
    return path


def _blocked(value: str) -> bool:
    path = _safe_path(value)
    parts = path.parts
    # These exact source trees are code, unlike any other runtime models directory.
    scoped_parts = parts[2:] if parts[:2] in {("deploy", "models"), ("app", "models")} else parts
    if any(
        part.lower() in BLOCKED_COMPONENTS or part.lower().startswith(".venv")
        for part in scoped_parts
    ):
        return True
    name = path.name.lower()
    if name.startswith(".env") or ".env." in name or name == ".sightindex-model-assets.lock":
        return not name.endswith(".env.example") and name != ".env.example"
    return any(name.endswith(suffix) for suffix in BLOCKED_SUFFIXES)


def _allowed(value: str, tracked: set[str]) -> bool:
    """Apply an explicit code/static-asset allowlist, independent of .gitignore."""
    path = _safe_path(value)
    if _blocked(value):
        return False
    if path.name.endswith(".env.example") or path.name == ".env.example":
        return value in tracked and (value in ROOT_FILES or value.startswith("deploy/"))
    if value in ROOT_FILES | DOCUMENTS | FRONTEND_FILES | OPERATOR_FILES:
        return True
    if len(path.parts) == 1:
        return bool(
            re.fullmatch(
                r"requirements(?:\.[A-Za-z0-9_-]+)?\.txt|README(?:\.[A-Za-z0-9_-]+)?\.md", value
            )
        )
    if path.parts[0] == "app":
        return path.suffix == ".py"
    if value.startswith("frontend/dist/"):
        return path.suffix.lower() in DIST_EXTENSIONS or (
            path.suffix.lower() in {".txt", ".json"}
            and "frontend/public/" + value.removeprefix("frontend/dist/") in tracked
        )
    if value.startswith(("frontend/src/", "frontend/scripts/")):
        return path.suffix in FRONTEND_EXTENSIONS or (
            path.suffix.lower() in STATIC_EXTENSIONS and value in tracked
        )
    if value.startswith("frontend/public/"):
        return value in tracked and path.suffix.lower() in STATIC_EXTENSIONS | {
            ".txt",
            ".json",
            ".html",
        }
    if value.startswith(("deploy/containers/", "deploy/models/", "deploy/rtx5090/")):
        return path.suffix in CODE_EXTENSIONS or path.name.startswith("Dockerfile")
    return False


def _tracked_paths(source: Path) -> set[str]:
    """Read Git's index only to authorize repository-native images and templates."""
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "ls-files", "--cached", "-z"],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    return {item.decode("utf-8", errors="strict") for item in result.stdout.split(b"\0") if item}


def _walk(source: Path, relative: str) -> list[Path]:
    """Traverse one source tree without following links or entering runtime state."""
    candidate = source / relative
    if not candidate.exists() and not candidate.is_symlink():
        return []
    info = candidate.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise BundleError("symlink in selected source tree")
    if not stat.S_ISDIR(info.st_mode):
        raise BundleError("selected source tree is not a directory")
    files: list[Path] = []
    for child in sorted(candidate.iterdir()):
        child_info = child.lstat()
        child_relative = child.relative_to(source).as_posix()
        if stat.S_ISLNK(child_info.st_mode):
            raise BundleError("symlink in selected source tree")
        if _blocked(child_relative):
            continue
        if stat.S_ISDIR(child_info.st_mode):
            files.extend(_walk(source, child_relative))
        elif stat.S_ISREG(child_info.st_mode):
            files.append(child)
        else:
            raise BundleError("non-regular file in selected source tree")
    return files


def _snapshot(source: Path) -> list[_Snapshot]:
    """Capture allowlisted file bytes once, never reopening during tar creation."""
    tracked = _tracked_paths(source)
    candidates: set[Path] = set()
    for child in source.iterdir():
        if _allowed(child.name, tracked):
            candidates.add(child)
    for value in DOCUMENTS | FRONTEND_FILES | OPERATOR_FILES:
        candidate = source / value
        if candidate.exists() or candidate.is_symlink():
            candidates.add(candidate)
    for tree in TREES:
        candidates.update(_walk(source, tree))
    snapshots: list[_Snapshot] = []
    total = 0
    for candidate in sorted(candidates):
        relative = candidate.relative_to(source).as_posix()
        if not _allowed(relative, tracked):
            continue
        # Hold directory descriptors while traversing: no ancestor or final-path
        # symlink can race the earlier inventory scan and escape the source root.
        with ExitStack() as descriptors:
            directory = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            descriptors.callback(os.close, directory)
            for part in PurePosixPath(relative).parts[:-1]:
                directory = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
                descriptors.callback(os.close, directory)
            descriptor = os.open(candidate.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            with os.fdopen(descriptor, "rb") as handle:
                before = os.fstat(handle.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
                    raise BundleError("invalid or oversized source file")
                content = handle.read(MAX_FILE_BYTES + 1)
                after = os.fstat(handle.fileno())
        if (
            len(content) != before.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_size != after.st_size
        ):
            raise BundleError("source file changed while building bundle")
        total += len(content)
        if total > MAX_BUNDLE_BYTES or len(snapshots) >= MAX_FILE_COUNT:
            raise BundleError("source bundle exceeds safety limits")
        snapshots.append(_Snapshot(relative, content, 0o755 if before.st_mode & 0o111 else 0o644))
    names = {item.path for item in snapshots}
    if (
        not {
            "main.py",
            "app/__init__.py",
            "app/models/media.py",
            "app/db/session.py",
            "frontend/dist/index.html",
        }
        <= names
    ):
        raise BundleError("bundle requires app, main.py and built frontend/dist/index.html")
    return snapshots


def _tar_member(archive: tarfile.TarFile, name: str, content: bytes, mode: int) -> None:
    """Write one deterministic regular file with no owner, link, or PAX metadata."""
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mode = mode
    info.mtime = 0
    info.uid = info.gid = 0
    archive.addfile(info, io.BytesIO(content))


def build_bundle(source: Path, output: Path, revision: str) -> str:
    """Create a private code-only tar.gz without overwriting an existing path.

    Args:
        source: Repository root whose selected source files are to be captured.
        output: New archive path in an existing directory.
        revision: Explicit dirty release label, e.g. ``36a4adc-dirty-20261007``.

    Returns:
        SHA256 of the exact SOURCE_MANIFEST.json bytes stored in the archive.

    Raises:
        BundleError: An unsafe source, label, file, or output is encountered.
        FileExistsError: Output already exists, including a dangling symlink.
    """
    if not REVISION_PATTERN.fullmatch(revision):
        raise BundleError("revision must be a commit-prefix-dirty-release label")
    if source.is_symlink() or not source.is_dir():
        raise BundleError("source must be a real repository directory")
    source = source.resolve(strict=True)
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise BundleError("output parent must be an existing real directory")
    output = output.parent.resolve(strict=True) / output.name
    if output.exists() or output.is_symlink():
        raise FileExistsError("bundle output already exists")
    if output.is_relative_to(source):
        raise BundleError("bundle output must be outside the source tree")
    snapshots = _snapshot(source)
    manifest = {
        "schema_version": 1,
        "revision": revision,
        "dirty": True,
        "protected_paths": list(PROTECTED_PATHS),
        "entries": [item.entry() for item in snapshots],
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    fd, temporary_name = tempfile.mkstemp(prefix=".sightindex-bundle-", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
                ) as archive:
                    for item in snapshots:
                        _tar_member(archive, item.path, item.content, item.mode)
                    _tar_member(archive, MANIFEST_NAME, manifest_bytes, 0o644)
            raw.flush()
            os.fsync(raw.fileno())
        verify_bundle(temporary, expected_manifest_sha256=manifest_sha256)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_sha256


def verify_bundle(bundle: Path, *, expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Read and verify the exact archive inventory; never extract or mutate it."""
    try:
        if bundle.is_symlink() or not bundle.is_file():
            raise BundleError("archive must be a regular file")
        with tarfile.open(bundle, mode="r:gz") as archive:
            names: set[str] = set()
            archive_bytes = 0
            for member in archive:
                _safe_path(member.name)
                if (
                    not member.isfile()
                    or member.linkname
                    or member.pax_headers
                    or not 0 <= member.size <= MAX_FILE_BYTES
                ):
                    raise BundleError("archive must contain bounded regular files only")
                if member.name in names or len(names) >= MAX_FILE_COUNT + 1:
                    raise BundleError("duplicate members or oversized archive inventory")
                names.add(member.name)
                archive_bytes += member.size
                if archive_bytes > MAX_BUNDLE_BYTES + MAX_FILE_BYTES:
                    raise BundleError("archive exceeds safety limits")
            manifest_member = archive.getmember(MANIFEST_NAME)
            if manifest_member.mode != 0o644:
                raise BundleError("manifest must have ordinary non-executable permissions")
            manifest_handle = archive.extractfile(manifest_member)
            if manifest_handle is None:
                raise BundleError("missing manifest")
            manifest_bytes = manifest_handle.read(MAX_FILE_BYTES + 1)
            if (
                expected_manifest_sha256 is not None
                and hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256
            ):
                raise BundleError("manifest SHA256 mismatch")
            manifest = json.loads(manifest_bytes)
            if (
                not isinstance(manifest, dict)
                or type(manifest.get("schema_version")) is not int
                or manifest.get("schema_version") != 1
                or manifest.get("dirty") is not True
                or not isinstance(manifest.get("revision"), str)
                or not REVISION_PATTERN.fullmatch(manifest["revision"])
                or manifest.get("protected_paths") != list(PROTECTED_PATHS)
                or not isinstance(manifest.get("entries"), list)
            ):
                raise BundleError("invalid source manifest")
            entries: dict[str, dict[str, Any]] = {}
            for entry in manifest["entries"]:
                if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "bytes", "mode"}:
                    raise BundleError("invalid manifest entry")
                value = entry["path"]
                if not isinstance(value, str) or value == MANIFEST_NAME or value in entries:
                    raise BundleError("invalid or duplicate manifest path")
                _safe_path(value)
                # A transferred manifest cannot grant itself permission to carry runtime files.
                if not _allowed(value, {value}):
                    raise BundleError("manifest contains a disallowed path")
                if (
                    type(entry["bytes"]) is not int
                    or not 0 <= entry["bytes"] <= MAX_FILE_BYTES
                    or not isinstance(entry["sha256"], str)
                    or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
                    or not isinstance(entry["mode"], str)
                    or entry["mode"] not in {"0644", "0755"}
                ):
                    raise BundleError("invalid manifest identity")
                entries[value] = entry
            if set(names) != set(entries) | {MANIFEST_NAME}:
                raise BundleError("archive inventory differs from manifest")
            if not {
                "main.py",
                "app/__init__.py",
                "app/models/media.py",
                "app/db/session.py",
                "frontend/dist/index.html",
            } <= set(entries):
                raise BundleError("archive is missing required application files")
            total = 0
            for value, entry in entries.items():
                member = archive.getmember(value)
                handle = archive.extractfile(member)
                if handle is None:
                    raise BundleError("missing archive member")
                content = handle.read(MAX_FILE_BYTES + 1)
                total += len(content)
                if (
                    len(content) != entry["bytes"]
                    or member.size != entry["bytes"]
                    or member.mode != int(entry["mode"], 8)
                    or hashlib.sha256(content).hexdigest() != entry["sha256"]
                    or total > MAX_BUNDLE_BYTES
                ):
                    raise BundleError("archive member identity mismatch")
            return manifest
    except (OSError, tarfile.TarError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BundleError("source bundle could not be verified") from exc


def main(argv: list[str] | None = None) -> int:
    """Build or read-only verify an archive, printing bounded metadata only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            if args.revision is not None:
                parser.error("--revision is only valid with --output")
            manifest = verify_bundle(args.verify, expected_manifest_sha256=args.manifest_sha256)
            print(
                json.dumps(
                    {
                        "verified": True,
                        "revision": manifest["revision"],
                        "file_count": len(manifest["entries"]),
                    }
                )
            )
        else:
            if args.revision is None or args.manifest_sha256 is not None:
                parser.error("--output requires --revision and does not accept --manifest-sha256")
            digest = build_bundle(args.source, args.output, args.revision)
            manifest = verify_bundle(args.output, expected_manifest_sha256=digest)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "manifest_sha256": digest,
                        "revision": args.revision,
                        "file_count": len(manifest["entries"]),
                    }
                )
            )
    except (BundleError, OSError):
        print(json.dumps({"error": "unsafe_or_invalid_source_bundle"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
