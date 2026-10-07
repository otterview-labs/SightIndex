"""Prepare explicitly reviewed model assets without guessing hashes or fetching manifests.

Only the standard library is used. ``check`` reads files only; ``prepare`` requires
an offline source directory or explicit HTTPS download permission. Results contain
bounded identifiers and safe codes, never upstream URLs or exception response bodies.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal, Protocol, cast
from urllib import request
from urllib.parse import urlsplit

CHUNK_BYTES = 1024 * 1024
MARGIN_BYTES = 64 * 1024 * 1024
LOCK_NAME = ".sightindex-model-assets.lock"
LOCK_MARKER = b"SightIndex-Model-Assets-Lock/2\n"
QUARANTINE_NAME = ".sightindex-model-assets-quarantine"
MANIFEST_LIMIT_BYTES = 4 * 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
PATH_COMPONENT = re.compile(r"[A-Za-z0-9_.-]{1,255}\Z")
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
Mode = Literal["check", "prepare", "diagnose", "recover"]
Status = Literal[
    "verified", "missing", "mismatch", "reused", "prepared", "failed", "preserved", "quarantined"
]


class AssetError(ValueError):
    """A bounded diagnostic code safe to serialize without reproducing input."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _https_url(value: str) -> None:
    """Require HTTPS and refuse inline credentials or malformed URLs."""
    try:
        parsed = urlsplit(value)
        valid = (
            len(value) <= 8192
            and parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
            and (parsed.port is None or 1 <= parsed.port <= 65535)
            and not any(ord(character) < 33 for character in value)
        )
    except ValueError:
        valid = False
    if not valid:
        raise AssetError("manifest.invalid_https_url")


def _relative_path(value: str) -> None:
    """Require a portable relative layout with no traversal or internal lock names."""
    path = PurePosixPath(value)
    if (
        not value
        or not path.parts
        or len(value) > 2048
        or path.is_absolute()
        or str(path) != value
        or len(path.name) + len(".partial") > 255
        or any(
            part.casefold() in {".", "..", LOCK_NAME, QUARANTINE_NAME}
            or not PATH_COMPONENT.fullmatch(part)
            for part in path.parts
        )
    ):
        raise AssetError("manifest.invalid_path")


@dataclass(frozen=True, slots=True)
class Artifact:
    """One file with operator-approved identity and integrity metadata."""

    id: str
    path: str
    sha256: str
    size_bytes: int
    model: str
    revision: str
    source_url: str | None = field(default=None, repr=False)
    terms_url: str | None = field(default=None, repr=False)
    license: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not IDENTIFIER.fullmatch(self.id):
            raise AssetError("manifest.invalid_id")
        if not isinstance(self.path, str):
            raise AssetError("manifest.invalid_path")
        _relative_path(self.path)
        if not isinstance(self.sha256, str) or not SHA256.fullmatch(self.sha256):
            raise AssetError("manifest.invalid_sha256")
        object.__setattr__(self, "sha256", self.sha256.lower())
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise AssetError("manifest.invalid_size")
        for value in (self.model, self.revision):
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 1024
                or any(ord(character) < 32 for character in value)
            ):
                raise AssetError("manifest.invalid_metadata")
        if self.license is not None and (
            not isinstance(self.license, str)
            or not self.license.strip()
            or len(self.license) > 1024
            or any(ord(character) < 32 for character in self.license)
        ):
            raise AssetError("manifest.invalid_license")
        for value in (self.source_url, self.terms_url):
            if value is not None:
                if not isinstance(value, str):
                    raise AssetError("manifest.invalid_https_url")
                _https_url(value)


@dataclass(frozen=True, slots=True)
class Manifest:
    """Versioned artifacts; constructing a filtered tuple keeps validation intact."""

    version: int
    artifacts: tuple[Artifact, ...]

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise AssetError("manifest.unsupported_version")
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            raise AssetError("manifest.empty_artifacts")
        identifiers: set[str] = set()
        paths: set[str] = set()
        for artifact in self.artifacts:
            if not isinstance(artifact, Artifact):
                raise AssetError("manifest.invalid_artifact")
            if artifact.id in identifiers:
                raise AssetError("manifest.duplicate_id")
            path = artifact.path.casefold()
            if path in paths:
                raise AssetError("manifest.duplicate_path")
            identifiers.add(artifact.id)
            paths.add(path)
        _validate_file_layout(tuple(Path(artifact.path) for artifact in self.artifacts))


@dataclass(frozen=True, slots=True)
class AssetResult:
    """An artifact result without URL, upstream response, or metadata secrets."""

    id: str
    path: str
    status: Status
    code: str

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-ready, bounded result."""
        return {"id": self.id, "path": self.path, "status": self.status, "code": self.code}


@dataclass(frozen=True, slots=True)
class AssetReport:
    """Structured aggregate used by deployment integration and the CLI."""

    mode: Mode
    ok: bool
    results: tuple[AssetResult, ...] = ()
    error: str | None = None
    lock: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Serialize only safe diagnostics."""
        output: dict[str, object] = {
            "mode": self.mode,
            "ok": self.ok,
            "results": [result.as_dict() for result in self.results],
        }
        if self.error:
            output["error"] = {"code": self.error}
        if self.lock:
            output["lock"] = {"code": self.lock}
        return output


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys instead of silently replacing approved metadata."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AssetError("manifest.duplicate_key")
        result[key] = value
    return result


def load_manifest(path: Path) -> Manifest:
    """Read a local reviewed manifest; never fetch or infer one.

    Raises:
        AssetError: The schema, file safety, or integrity metadata is invalid.
    """
    try:
        with _parent_directory(path, create=False) as (directory, name):
            with _read_file(directory, name) as source:
                raw = source.read(MANIFEST_LIMIT_BYTES + 1)
        if len(raw) > MANIFEST_LIMIT_BYTES:
            raise AssetError("manifest.too_large")
        payload: object = json.loads(raw, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssetError("manifest.unreadable") from exc
    if not isinstance(payload, dict) or set(payload) != {"version", "artifacts"}:
        raise AssetError("manifest.invalid_fields")
    entries = payload["artifacts"]
    if not isinstance(entries, list):
        raise AssetError("manifest.invalid_artifacts")
    required = {"id", "path", "sha256", "size_bytes", "model", "revision"}
    optional = {"source_url", "terms_url", "license"}
    artifacts: list[Artifact] = []
    for entry in entries:
        if not isinstance(entry, dict) or not required <= set(entry) <= required | optional:
            raise AssetError("manifest.invalid_artifact_fields")
        # Dataclass validation checks runtime types before fields are used.
        artifacts.append(Artifact(**entry))
    return Manifest(payload["version"], tuple(artifacts))


def _validate_file_layout(paths: tuple[Path, ...]) -> None:
    """Reject file/parent collisions and artifact/partial cache aliases."""
    names = {str(path).casefold() for path in paths}
    if len(names) != len(paths):
        raise AssetError("manifest.duplicate_destination")
    partials = {str(path.with_name(path.name + ".partial")).casefold() for path in paths}
    if names & partials:
        raise AssetError("manifest.path_collision")
    for path in paths:
        if any(str(parent).casefold() in names | partials for parent in path.parents):
            raise AssetError("manifest.path_collision")


def _absolute(path: Path) -> Path:
    """Normalize lexical paths without resolving symlinks or allowing traversal."""
    if ".." in path.parts:
        raise AssetError("filesystem.unsafe_path")
    return path.absolute()


@contextmanager
def _directory(path: Path, *, create: bool) -> Iterator[int]:
    """Walk directories by file descriptor, refusing symlinks at every component."""
    absolute = _absolute(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=directory)
            except FileNotFoundError:
                if not create:
                    raise
                created = False
                try:
                    os.mkdir(component, 0o755, dir_fd=directory)
                    created = True
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=directory)
                if created:
                    try:
                        # Installer umask 077 must not make new model directories
                        # untraversable by the non-root service. Existing paths,
                        # including a mkdir race winner, keep operator permissions.
                        os.fchmod(child, 0o755)
                    except OSError:
                        os.close(child)
                        raise
            except OSError as exc:
                raise AssetError("filesystem.unsafe_directory") from exc
            os.close(directory)
            directory = child
        yield directory
    finally:
        os.close(directory)


@contextmanager
def _parent_directory(path: Path, *, create: bool) -> Iterator[tuple[int, str]]:
    """Keep the containing directory pinned while handling a file."""
    absolute = _absolute(path)
    if not absolute.name:
        raise AssetError("filesystem.unsafe_path")
    with _directory(absolute.parent, create=create) as directory:
        yield directory, absolute.name


def _file_stat(directory: int, name: str) -> os.stat_result | None:
    """Inspect a regular file without following symlinks; hardlinks are refused."""
    try:
        value = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise AssetError("filesystem.unsafe_file")
    return value


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare identity and content metadata to detect replacement during reads."""
    return (
        left.st_dev,
        left.st_ino,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


@contextmanager
def _read_file(directory: int, name: str) -> Iterator[BinaryIO]:
    """Open the inspected single-link file without a symlink replacement race."""
    before = _file_stat(directory, name)
    if before is None:
        raise FileNotFoundError
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as source:
        opened = os.fstat(source.fileno())
        if opened.st_nlink != 1 or not _same_file(before, opened):
            raise AssetError("filesystem.file_changed")
        yield source
        after = _file_stat(directory, name)
        if (
            after is None
            or not _same_file(opened, after)
            or not _same_file(opened, os.fstat(source.fileno()))
        ):
            raise AssetError("filesystem.file_changed")


def _digest(source: BinaryIO) -> str:
    """Hash incrementally, without loading model weights into memory."""
    checksum = hashlib.sha256()
    while chunk := source.read(CHUNK_BYTES):
        checksum.update(chunk)
    return checksum.hexdigest()


def _verify_file(directory: int, name: str, artifact: Artifact) -> None:
    """Require exact size and SHA-256 on a safe, stable regular file."""
    value = _file_stat(directory, name)
    if value is None:
        raise AssetError("asset.missing")
    if value.st_size != artifact.size_bytes:
        raise AssetError("asset.size_mismatch")
    with _read_file(directory, name) as source:
        if _digest(source) != artifact.sha256.lower():
            raise AssetError("asset.sha256_mismatch")


def _plans(
    manifest: Manifest,
    destination: Path,
    destination_for: Callable[[Artifact], Path] | None,
) -> tuple[tuple[Artifact, Path], ...]:
    """Resolve only preselected artifacts; mappings belong to the deployment caller."""
    base = _absolute(destination)
    if base == Path(base.anchor):
        raise AssetError("filesystem.destination_is_root")
    plans = tuple(
        (
            artifact,
            _absolute(destination_for(artifact) if destination_for else base / artifact.path),
        )
        for artifact in manifest.artifacts
    )
    _validate_file_layout(tuple(path for _artifact, path in plans))
    if any(
        {part.casefold() for part in path.parts} & {LOCK_NAME, QUARANTINE_NAME}
        for _artifact, path in plans
    ):
        raise AssetError("manifest.path_collision")
    return plans


def _inspect(artifact: Artifact, target: Path, mode: Mode) -> tuple[AssetResult, int]:
    """Read existing target and partial safety, returning remaining required bytes."""
    try:
        with _parent_directory(target, create=False) as (directory, name):
            partial = _file_stat(directory, name + ".partial")
            if partial is not None and partial.st_size > artifact.size_bytes:
                raise AssetError("asset.partial_size_invalid")
            if _file_stat(directory, name) is None:
                remaining = artifact.size_bytes - (partial.st_size if partial else 0)
                if remaining == 0:
                    _verify_file(directory, name + ".partial", artifact)
                return AssetResult(
                    artifact.id, artifact.path, "missing", "asset.missing"
                ), remaining
            _verify_file(directory, name, artifact)
        status: Status = "verified" if mode == "check" else "reused"
        return AssetResult(artifact.id, artifact.path, status, "asset.verified"), 0
    except FileNotFoundError:
        return AssetResult(
            artifact.id, artifact.path, "missing", "asset.missing"
        ), artifact.size_bytes
    except AssetError as exc:
        status = "mismatch" if exc.code.startswith("asset.") else "failed"
        return AssetResult(artifact.id, artifact.path, status, exc.code), 0
    except OSError:
        return AssetResult(artifact.id, artifact.path, "failed", "filesystem.unreadable"), 0


def check_assets(
    manifest: Manifest,
    destination: Path,
    *,
    destination_for: Callable[[Artifact], Path] | None = None,
) -> AssetReport:
    """Check all selected existing files without mkdir, lock creation, or network."""
    try:
        plans = _plans(manifest, destination, destination_for)
    except AssetError as exc:
        return AssetReport("check", False, error=exc.code)
    results = tuple(_inspect(artifact, path, "check")[0] for artifact, path in plans)
    return AssetReport("check", all(result.status == "verified" for result in results), results)


@contextmanager
def _package_lock(destination: Path) -> Iterator[None]:
    """Hold a persistent POSIX kernel lock; legacy lock files are never guessed stale.

    Keeping the inode after release prevents two preparers from locking different
    files under the same name. The kernel releases flock on exit, including SIGKILL.
    """
    with _directory(destination, create=True) as directory:
        try:
            descriptor = os.open(
                LOCK_NAME,
                os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
            created = True
        except FileExistsError:
            _file_stat(directory, LOCK_NAME)
            descriptor = os.open(LOCK_NAME, os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory)
            created = False
        try:
            _flock(descriptor)
            _lock_identity(directory, descriptor)
            if created:
                os.write(descriptor, LOCK_MARKER)
                os.fsync(descriptor)
                os.fsync(directory)
            elif os.read(descriptor, len(LOCK_MARKER) + 1) != LOCK_MARKER:
                raise AssetError("prepare.package_locked")
            yield
        finally:
            os.close(descriptor)


def _flock(descriptor: int) -> None:
    """Refuse every active v2 lock without waiting or attempting an override."""
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise AssetError("prepare.package_locked") from exc


def _lock_identity(directory: int, descriptor: int) -> os.stat_result:
    """Require the held descriptor to still be the single-link named lock inode."""
    opened = os.fstat(descriptor)
    current = _file_stat(directory, LOCK_NAME)
    if current is None or not _same_file(opened, current):
        raise AssetError("filesystem.file_changed")
    return opened


def _lock_state(destination: Path) -> str:
    """Read and probe an existing lock only; never create or modify any path."""
    try:
        with _directory(destination, create=False) as directory:
            if _file_stat(directory, LOCK_NAME) is None:
                return "lock.absent"
            descriptor = os.open(LOCK_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            try:
                _lock_identity(directory, descriptor)
                try:
                    _flock(descriptor)
                except AssetError as exc:
                    if exc.code == "prepare.package_locked":
                        return "lock.active"
                    raise
                marker = os.read(descriptor, len(LOCK_MARKER) + 1)
                _lock_identity(directory, descriptor)
                if marker == LOCK_MARKER:
                    return "lock.released"
                return "lock.legacy_empty" if not marker else "lock.unknown_format"
            finally:
                os.close(descriptor)
    except FileNotFoundError:
        return "lock.absent"


def _partial_state(artifact: Artifact, target: Path) -> AssetResult:
    """Inspect final/partial files; a short prefix is unverified, not proven corrupt."""
    checking_partial = False
    try:
        with _parent_directory(target, create=False) as (directory, name):
            partial = _file_stat(directory, name + ".partial")
            if _file_stat(directory, name) is not None:
                _verify_file(directory, name, artifact)
                return AssetResult(artifact.id, artifact.path, "verified", "asset.verified")
            checking_partial = True
            if partial is None:
                return AssetResult(artifact.id, artifact.path, "missing", "partial.absent")
            if partial.st_size < artifact.size_bytes:
                return AssetResult(
                    artifact.id, artifact.path, "preserved", "partial.unverified_prefix"
                )
            _verify_file(directory, name + ".partial", artifact)
            return AssetResult(artifact.id, artifact.path, "preserved", "partial.verified")
    except FileNotFoundError:
        return AssetResult(artifact.id, artifact.path, "missing", "partial.absent")
    except AssetError as exc:
        prefix = "partial." if checking_partial and exc.code.startswith("asset.") else ""
        return AssetResult(artifact.id, artifact.path, "mismatch", prefix + exc.code)


def diagnose_assets(
    manifest: Manifest,
    destination: Path,
    *,
    destination_for: Callable[[Artifact], Path] | None = None,
) -> AssetReport:
    """Diagnose selected interrupted assets read-only; this is not model readiness."""
    try:
        plans = _plans(manifest, destination, destination_for)
        lock = _lock_state(destination)
        results = tuple(_partial_state(artifact, path) for artifact, path in plans)
        ok = lock in {"lock.absent", "lock.released"} and not any(
            result.status == "mismatch" for result in results
        )
        return AssetReport("diagnose", ok, results, lock=lock)
    except AssetError as exc:
        return AssetReport("diagnose", False, error=exc.code)
    except OSError:
        return AssetReport("diagnose", False, error="filesystem.unreadable")


@contextmanager
def _quarantine_directory(directory: int) -> Iterator[int]:
    """Create a private unique quarantine on the same filesystem as the original."""
    try:
        os.mkdir(QUARANTINE_NAME, 0o700, dir_fd=directory)
    except FileExistsError:
        pass
    parent = os.open(
        QUARANTINE_NAME, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
    )
    try:
        info = os.fstat(parent)
        if info.st_mode & 0o777 != 0o700 or info.st_uid != os.geteuid():
            raise AssetError("recovery.quarantine_not_private")
        run = uuid.uuid4().hex
        os.mkdir(run, 0o700, dir_fd=parent)
        quarantine = os.open(run, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            os.fsync(parent)
            yield quarantine
        finally:
            os.close(quarantine)
    finally:
        os.close(parent)


@contextmanager
def _recovery_lock(destination: Path) -> Iterator[str]:
    """Recover only an explicitly confirmed empty legacy lock, preserving its bytes.

    The legacy name remains present until replacement by the already-flocked v2
    inode, so old O_EXCL preparers cannot enter between isolation and migration.
    Active pre-v2 processes cannot be detected by flock: the operator must stop
    them before confirming recovery. Unknown lock contents are never overridden.
    """
    with _directory(destination, create=True) as directory:
        if _file_stat(directory, LOCK_NAME) is None:
            with _package_lock(destination):
                yield "lock.released"
            return
        descriptor = os.open(LOCK_NAME, os.O_RDWR | os.O_NOFOLLOW, dir_fd=directory)
        try:
            _flock(descriptor)
            identity = _lock_identity(directory, descriptor)
            marker = os.read(descriptor, len(LOCK_MARKER) + 1)
            if marker == LOCK_MARKER:
                yield "lock.released"
                return
            if marker:
                raise AssetError("recovery.unknown_lock_format")
            with _quarantine_directory(directory) as quarantine:
                replacement = os.open(
                    "lock-v2",
                    os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=quarantine,
                )
                try:
                    _flock(replacement)
                    os.write(replacement, LOCK_MARKER)
                    os.fsync(replacement)
                    _lock_identity(directory, descriptor)
                    # Empty legacy files carry no owner metadata. Preserve their
                    # exact bytes before replacement, rather than a hardlink: a
                    # crash between link and replace would leave a two-link lock
                    # that the safe-file policy correctly refuses on the next run.
                    archived = os.open(
                        "legacy.lock",
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=quarantine,
                    )
                    try:
                        os.fsync(archived)
                    finally:
                        os.close(archived)
                    _lock_identity(directory, descriptor)
                    if identity.st_size != 0:
                        raise AssetError("recovery.unknown_lock_format")
                    os.fsync(quarantine)
                    # The old name exists throughout, preventing legacy O_EXCL races.
                    os.replace("lock-v2", LOCK_NAME, src_dir_fd=quarantine, dst_dir_fd=directory)
                    _lock_identity(directory, replacement)
                    os.fsync(directory)
                    os.fsync(quarantine)
                    yield "lock.legacy_quarantined"
                finally:
                    os.close(replacement)
        finally:
            os.close(descriptor)


def _recover_partial(artifact: Artifact, target: Path, *, reset: bool) -> AssetResult:
    """Atomically move only one selected partial into its parent's private quarantine."""
    current = _partial_state(artifact, target)
    try:
        with _parent_directory(target, create=False) as (directory, name):
            final = _file_stat(directory, name)
            partial_name = name + ".partial"
            partial = _file_stat(directory, partial_name)
            corrupt = current.code in {
                "partial.asset.size_mismatch",
                "partial.asset.sha256_mismatch",
            }
            if partial is not None and final is None and (corrupt or reset):
                with _quarantine_directory(directory) as quarantine:
                    before = _file_stat(directory, partial_name)
                    if before is None or not _same_file(before, partial):
                        raise AssetError("filesystem.file_changed")
                    os.rename(
                        partial_name, partial_name, src_dir_fd=directory, dst_dir_fd=quarantine
                    )
                    isolated = _file_stat(quarantine, partial_name)
                    if isolated is None or (isolated.st_dev, isolated.st_ino) != (
                        partial.st_dev,
                        partial.st_ino,
                    ):
                        raise AssetError("filesystem.file_changed")
                    os.fsync(quarantine)
                    os.fsync(directory)
                current = AssetResult(
                    artifact.id, artifact.path, "quarantined", "partial.quarantined"
                )
    except FileNotFoundError as exc:
        if current.status != "missing":
            raise AssetError("filesystem.file_changed") from exc
    return current


def recover_assets(
    manifest: Manifest,
    destination: Path,
    *,
    confirm_no_active_preparation: bool = False,
    quarantine_partial_ids: frozenset[str] = frozenset(),
    destination_for: Callable[[Artifact], Path] | None = None,
) -> AssetReport:
    """Isolate interrupted state, never delete/replace a final weight or fetch bytes."""
    results: list[AssetResult] = []
    try:
        if not confirm_no_active_preparation:
            raise AssetError("recovery.confirm_no_active_preparation_required")
        plans = _plans(manifest, destination, destination_for)
        if quarantine_partial_ids - {artifact.id for artifact, _path in plans}:
            raise AssetError("recovery.unknown_partial_role")
        with _recovery_lock(destination) as lock:
            for artifact, target in plans:
                results.append(
                    _recover_partial(artifact, target, reset=artifact.id in quarantine_partial_ids)
                )
        return AssetReport(
            "recover",
            not any(result.status == "mismatch" for result in results),
            tuple(results),
            lock=lock,
        )
    except AssetError as exc:
        return AssetReport("recover", False, tuple(results), exc.code)
    except OSError:
        return AssetReport("recover", False, tuple(results), "filesystem.recovery_failed")


def _existing_directory(path: Path) -> Path:
    """Find an existing directory for disk accounting without creating anything."""
    candidate = _absolute(path)
    while not candidate.exists():
        candidate = candidate.parent
    with _directory(candidate, create=False):
        return candidate


def _disk_gate(
    plans: tuple[tuple[Artifact, Path], ...],
    remaining: tuple[int, ...],
    margin_bytes: int,
    free_bytes: Callable[[Path], int],
) -> None:
    """Gate remaining manifest bytes plus margin on every destination filesystem."""
    filesystems: dict[int, tuple[Path, int]] = {}
    for (_artifact, target), missing in zip(plans, remaining, strict=True):
        if missing == 0:
            continue
        ancestor = _existing_directory(target.parent)
        device = ancestor.stat().st_dev
        prior = filesystems.get(device, (ancestor, 0))
        filesystems[device] = (prior[0], prior[1] + missing)
    if any(free_bytes(path) < size + margin_bytes for path, size in filesystems.values()):
        raise AssetError("prepare.insufficient_disk")


def _space_required(
    plans: tuple[tuple[Artifact, Path], ...],
    inspection: tuple[tuple[AssetResult, int], ...],
    *,
    download: bool,
) -> tuple[int, ...]:
    """Reserve a full download: the server may ignore Range and restart with 200."""
    return tuple(
        artifact.size_bytes if download and remaining else remaining
        for (artifact, _target), (_result, remaining) in zip(plans, inspection, strict=True)
    )


@contextmanager
def _partial_file(directory: int, name: str) -> Iterator[BinaryIO]:
    """Open or create a private single-link partial cache without truncating it."""
    previous = _file_stat(directory, name)
    flags = os.O_RDWR | os.O_NOFOLLOW
    if previous is None:
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(name, flags, 0o600, dir_fd=directory)
    with os.fdopen(descriptor, "r+b") as partial:
        opened = os.fstat(partial.fileno())
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise AssetError("filesystem.unsafe_file")
        if previous is not None and not _same_file(previous, opened):
            raise AssetError("filesystem.file_changed")
        yield partial
        partial.flush()
        os.fsync(partial.fileno())
        current = _file_stat(directory, name)
        if current is None or current.st_ino != opened.st_ino or current.st_dev != opened.st_dev:
            raise AssetError("filesystem.file_changed")


def _copy_offline(artifact: Artifact, source_dir: Path, partial: BinaryIO) -> None:
    """Resume copying a safe local file with an exact matching cached prefix."""
    with _parent_directory(source_dir / artifact.path, create=False) as (directory, name):
        source_stat = _file_stat(directory, name)
        if source_stat is None or source_stat.st_size != artifact.size_bytes:
            raise AssetError("source.size_or_file_invalid")
        with _read_file(directory, name) as source:
            offset = os.fstat(partial.fileno()).st_size
            partial.seek(0)
            remaining = offset
            while remaining:
                length = min(CHUNK_BYTES, remaining)
                if source.read(length) != partial.read(length):
                    raise AssetError("source.partial_prefix_mismatch")
                remaining -= length
            partial.seek(offset)
            while chunk := source.read(CHUNK_BYTES):
                if partial.tell() + len(chunk) > artifact.size_bytes:
                    raise AssetError("source.size_or_file_invalid")
                partial.write(chunk)


class ResponseHeaders(Protocol):
    """The small read-only header interface used by strict HTTP mocks."""

    def get(self, key: str, default: str | None = None) -> str | None: ...


class HTTPResponse(Protocol):
    """A bounded streaming response, injectable without real network access."""

    status: int
    headers: ResponseHeaders

    def read(self, size: int = -1) -> bytes: ...
    def geturl(self) -> str: ...
    def close(self) -> None: ...


HTTPOpener = Callable[[request.Request, float], HTTPResponse]


class _HTTPSRedirect(request.HTTPRedirectHandler):
    """Allow HTTPS redirects while refusing downgrade or inline credentials."""

    def redirect_request(
        self,
        req: request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> request.Request | None:
        try:
            _https_url(newurl)
        except AssetError as exc:
            raise AssetError("download.unsafe_redirect") from exc
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_http(req: request.Request, timeout: float) -> HTTPResponse:
    """Open a HTTPS request without inheriting proxy credentials from the process."""
    opener = request.build_opener(request.ProxyHandler({}), _HTTPSRedirect())
    return cast(HTTPResponse, opener.open(req, timeout=timeout))


def _download(
    artifact: Artifact,
    partial: BinaryIO,
    opener: HTTPOpener,
    timeout_seconds: float,
    max_download_seconds: float,
) -> None:
    """Strictly validate Range responses before writing only the partial cache."""
    if not artifact.source_url:
        raise AssetError("download.source_url_missing")
    offset = os.fstat(partial.fileno()).st_size
    headers = {"User-Agent": "SightIndex-Model-Assets/1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = request.Request(artifact.source_url, headers=headers, method="GET")
    deadline = time.monotonic() + max_download_seconds
    try:
        with closing(opener(req, min(timeout_seconds, max_download_seconds))) as response:
            _https_url(response.geturl())
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise AssetError("download.invalid_encoding")
            if response.status == 206:
                content_range = response.headers.get("Content-Range", "") or ""
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if not match or tuple(map(int, match.groups())) != (
                    offset,
                    artifact.size_bytes - 1,
                    artifact.size_bytes,
                ):
                    raise AssetError("download.invalid_content_range")
            elif response.status == 200:
                offset = 0
            else:
                raise AssetError("download.invalid_http_status")
            expected = artifact.size_bytes - offset
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) != expected):
                raise AssetError("download.invalid_content_length")
            if response.status == 200:
                partial.truncate(0)
            partial.seek(offset)
            received = 0
            while True:
                if time.monotonic() >= deadline:
                    raise AssetError("download.deadline_exceeded")
                chunk = response.read(min(CHUNK_BYTES, expected - received + 1))
                if not chunk:
                    break
                if received + len(chunk) > expected:
                    raise AssetError("download.response_too_large")
                partial.write(chunk)
                received += len(chunk)
            if received != expected:
                raise AssetError("download.incomplete")
    except AssetError:
        raise
    except Exception as exc:
        # Arbitrary HTTP/SSL exceptions can include signed URLs or response bodies.
        raise AssetError("download.failed") from exc


def _promote(directory: int, name: str, artifact: Artifact) -> None:
    """Publish verified bytes atomically without replacing any raced destination."""
    partial = name + ".partial"
    before = _file_stat(directory, partial)
    if before is None:
        raise AssetError("filesystem.file_changed")
    descriptor = os.open(partial, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as source:
        verified = os.fstat(source.fileno())
        if not _same_file(before, verified) or verified.st_nlink != 1:
            raise AssetError("filesystem.file_changed")
        if verified.st_size != artifact.size_bytes:
            raise AssetError("asset.size_mismatch")
        if _digest(source) != artifact.sha256.lower():
            raise AssetError("asset.sha256_mismatch")
        current = _file_stat(directory, partial)
        if (
            current is None
            or not _same_file(verified, current)
            or not _same_file(verified, os.fstat(source.fileno()))
        ):
            raise AssetError("filesystem.file_changed")
        try:
            # link is an atomic no-replace publication, unlike rename/replace.
            os.link(
                partial, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False
            )
        except FileExistsError as exc:
            raise AssetError("prepare.destination_appeared") from exc
        linked = os.stat(name, dir_fd=directory, follow_symlinks=False)
        try:
            cache = os.stat(partial, dir_fd=directory, follow_symlinks=False)
            opened = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(linked.st_mode)
                or linked.st_nlink != 2
                or (linked.st_dev, linked.st_ino) != (verified.st_dev, verified.st_ino)
                or (cache.st_dev, cache.st_ino) != (verified.st_dev, verified.st_ino)
                or opened.st_nlink != 2
                or opened.st_size != verified.st_size
                or opened.st_mtime_ns != verified.st_mtime_ns
            ):
                raise AssetError("filesystem.file_changed")
            # Change permissions on the verified descriptor, never a raced path.
            os.fchmod(source.fileno(), 0o644)
            os.fsync(source.fileno())
            if not _unlink_owned(directory, partial, verified):
                raise AssetError("filesystem.file_changed")
            os.fsync(directory)
        except (AssetError, OSError):
            # Only remove the link just observed, never a replacement inode.
            _unlink_owned(directory, name, linked)
            raise


def _unlink_owned(directory: int, name: str, identity: os.stat_result) -> bool:
    """Remove a package entry only if it still identifies our own inode."""
    try:
        current = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
        return False
    os.unlink(name, dir_fd=directory)
    return True


def prepare_assets(
    manifest: Manifest,
    destination: Path,
    *,
    source_dir: Path | None = None,
    download: bool = False,
    destination_for: Callable[[Artifact], Path] | None = None,
    opener: HTTPOpener | None = None,
    margin_bytes: int = MARGIN_BYTES,
    free_bytes: Callable[[Path], int] | None = None,
    timeout_seconds: float = 30.0,
    max_download_seconds: float = 3600.0,
) -> AssetReport:
    """Prepare approved assets with offline copy or explicit bounded HTTPS download.

    Destination mappings are for trusted deployment callers with preselected roles;
    the CLI always uses ``destination/artifact.path``. Filesystem safety is checked
    for mapped parents as well. A mismatch never overwrites an existing final file.
    """
    results: list[AssetResult] = []
    try:
        if (source_dir is None) == (not download):
            raise AssetError("prepare.choose_one_source")
        if type(margin_bytes) is not int or margin_bytes < 0:
            raise AssetError("prepare.invalid_margin")
        if any(
            not math.isfinite(value) or not 0 < value <= 86400
            for value in (timeout_seconds, max_download_seconds)
        ):
            raise AssetError("prepare.invalid_timeout")
        plans = _plans(manifest, destination, destination_for)
        inspection = tuple(_inspect(artifact, path, "prepare") for artifact, path in plans)
        results = [result for result, _remaining in inspection]
        if any(result.status in {"mismatch", "failed"} for result in results):
            return AssetReport("prepare", False, tuple(results), "prepare.existing_file_invalid")
        if all(result.status == "reused" for result in results):
            return AssetReport("prepare", True, tuple(results))
        if source_dir is not None:
            with _directory(source_dir, create=False):
                pass
        elif any(
            remaining and not artifact.source_url
            for (artifact, _), (_, remaining) in zip(plans, inspection, strict=True)
        ):
            raise AssetError("download.source_url_missing")
        free = free_bytes or (lambda path: shutil.disk_usage(path).free)
        _disk_gate(plans, _space_required(plans, inspection, download=download), margin_bytes, free)
        with _package_lock(destination):
            # Re-inspect under the package lock; another prepare may have completed.
            inspection = tuple(_inspect(artifact, path, "prepare") for artifact, path in plans)
            results = [result for result, _remaining in inspection]
            if any(result.status in {"mismatch", "failed"} for result in results):
                return AssetReport(
                    "prepare", False, tuple(results), "prepare.existing_file_invalid"
                )
            _disk_gate(
                plans, _space_required(plans, inspection, download=download), margin_bytes, free
            )
            for index, (artifact, target) in enumerate(plans):
                if results[index].status == "reused":
                    continue
                try:
                    with _parent_directory(target, create=True) as (directory, name):
                        if _file_stat(directory, name) is not None:
                            raise AssetError("prepare.destination_appeared")
                        with _partial_file(directory, name + ".partial") as partial:
                            if os.fstat(partial.fileno()).st_size < artifact.size_bytes:
                                if source_dir is not None:
                                    _copy_offline(artifact, source_dir, partial)
                                else:
                                    _download(
                                        artifact,
                                        partial,
                                        opener or _open_http,
                                        timeout_seconds,
                                        max_download_seconds,
                                    )
                        _promote(directory, name, artifact)
                    results[index] = AssetResult(
                        artifact.id, artifact.path, "prepared", "asset.verified"
                    )
                except AssetError as exc:
                    results[index] = AssetResult(artifact.id, artifact.path, "failed", exc.code)
                except OSError:
                    results[index] = AssetResult(
                        artifact.id, artifact.path, "failed", "filesystem.prepare_failed"
                    )
        return AssetReport(
            "prepare",
            all(result.status in {"reused", "prepared"} for result in results),
            tuple(results),
        )
    except AssetError as exc:
        return AssetReport("prepare", False, tuple(results), exc.code)
    except OSError:
        return AssetReport("prepare", False, tuple(results), "filesystem.prepare_failed")


class _SafeParser(argparse.ArgumentParser):
    """Do not reproduce arbitrary CLI values in parser error diagnostics."""

    def error(self, message: str) -> None:
        raise AssetError("cli.invalid_arguments")


def main(argv: Sequence[str] | None = None) -> int:
    """Print structured safe results; help has no filesystem or network effects."""
    parser = _SafeParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--mode", choices=("check", "prepare"), required=True)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--source-dir", type=Path, help="offline bundle in manifest layout")
    sources.add_argument("--download", action="store_true", help="explicitly permit HTTPS GETs")
    mode: Mode = "check"
    try:
        arguments = parser.parse_args(argv)
        mode = cast(Mode, arguments.mode)
        if mode == "check" and (arguments.source_dir is not None or arguments.download):
            raise AssetError("check.source_flags_forbidden")
        manifest = load_manifest(arguments.manifest)
        if mode == "check":
            report = check_assets(manifest, arguments.destination)
        else:
            report = prepare_assets(
                manifest,
                arguments.destination,
                source_dir=arguments.source_dir,
                download=arguments.download,
            )
    except AssetError as exc:
        report = AssetReport(mode, False, error=exc.code)
    except OSError:
        report = AssetReport(mode, False, error="filesystem.unreadable")
    print(json.dumps(report.as_dict(), sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
