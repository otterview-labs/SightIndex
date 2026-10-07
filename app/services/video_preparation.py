"""Bounded, opt-in CPU preparation of newly uploaded local videos only.

No GET endpoint calls this module, no historical media is scanned, and no external input URL
is accepted. Frame extraction must use the returned playback path to keep the same time axis.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.settings import Settings

VERSION = "browser-v1"
_FORMATS = "mov,matroska,webm,avi,mpeg,mpegts"
_MAX_PROBE_BYTES = 64 * 1024


class VideoPreparationError(RuntimeError):
    """A finite diagnostic code, never a command line or arbitrary decoder stderr."""

    def __init__(self, code: str, status_code: int = 422) -> None:
        self.code = code
        self.status_code = status_code
        self.video_id: uuid.UUID | None = None
        super().__init__(f"Video preparation failed: {code}")

    def api_detail(self) -> dict[str, object]:
        return {"code": self.code, "video_id": str(self.video_id) if self.video_id else None}


@dataclass(frozen=True, slots=True)
class VideoInfo:
    codec: str
    pixel_format: str
    width: int
    height: int
    duration_seconds: float
    has_audio: bool
    audio_codec: str | None


@dataclass(frozen=True, slots=True)
class PreparedVideo:
    path: Path
    url: str
    original_sha256: str
    status: str
    info: VideoInfo


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def _binary(name: str) -> str:
    if not name or any(character in name for character in "\x00\r\n"):
        raise VideoPreparationError("tool_unavailable", 503)
    resolved = shutil.which(name)
    if resolved is None:
        raise VideoPreparationError("tool_unavailable", 503)
    path = Path(resolved).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise VideoPreparationError("tool_unavailable", 503)
    return str(path)


def _owned_source(path: Path, settings: Settings) -> os.stat_result:
    root = settings.videos_dir
    try:
        if root.is_symlink() or path.parent != root or path.is_symlink():
            raise VideoPreparationError("invalid_source", 400)
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise VideoPreparationError("invalid_source", 400)
    except OSError:
        raise VideoPreparationError("invalid_source", 400) from None
    if info.st_size == 0:
        raise VideoPreparationError("invalid_video", 400)
    if info.st_size > settings.upload_video_max_bytes:
        raise VideoPreparationError("input_limit", 413)
    # Reject playlists/XML/concat before invoking a demuxer, regardless of claimed extension.
    with path.open("rb") as stream:
        header = stream.read(384)
    allowed = (
        (header[:4] == b"RIFF" and header[8:12] == b"AVI ")
        or header[:4] == b"\x1aE\xdf\xa3"
        or header[4:8] in {b"ftyp", b"moov", b"mdat", b"wide"}
        or header[:4] in {b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"}
        or (len(header) >= 377 and header[0] == header[188] == header[376] == 0x47)
    )
    if not allowed:
        raise VideoPreparationError("unsupported_container", 415)
    return info


def _unchanged(path: Path, before: os.stat_result) -> None:
    after = path.stat()
    if (
        path.is_symlink()
        or after.st_ino != before.st_ino
        or after.st_dev != before.st_dev
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_nlink != 1
    ):
        raise VideoPreparationError("source_changed")


@contextlib.contextmanager
def _preparation_lock(settings: Settings) -> Iterator[Path]:
    try:
        import fcntl
    except ImportError:
        raise VideoPreparationError("locking_unavailable", 503) from None
    directory = settings.data_dir / ".video-preparation"
    if directory.is_symlink():
        raise VideoPreparationError("invalid_work_directory", 503)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory_info = directory.stat()
    if directory_info.st_uid != os.geteuid() or directory_info.st_mode & 0o022:
        raise VideoPreparationError("invalid_work_directory", 503)
    lock_path = directory / "prepare.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError:
        raise VideoPreparationError("invalid_lock", 503) from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise VideoPreparationError("invalid_lock", 503)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise VideoPreparationError("transcode_busy", 503) from None
        yield directory
    finally:
        # Kernel releases flock on close/process death. Never delete a shared lock inode.
        os.close(descriptor)


def _disk_check(settings: Settings, *, reserve_output: bool = True) -> None:
    reserve = settings.video_preparation_max_output_bytes if reserve_output else 0
    if (
        shutil.disk_usage(settings.videos_dir).free
        < settings.video_preparation_min_free_bytes + reserve
    ):
        raise VideoPreparationError("disk_limit", 507)


def _run(
    arguments: list[str],
    *,
    timeout: float,
    failure_code: str,
    settings: Settings,
    output: Path | None = None,
) -> bytes:
    """Use a new process group, bounded stdout and local-only, credential-free environment."""
    with tempfile.TemporaryFile() as stdout:
        try:
            process = subprocess.Popen(
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=subprocess.DEVNULL,
                env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
                start_new_session=True,
                close_fds=True,
            )
        except OSError:
            raise VideoPreparationError("tool_unavailable", 503) from None
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    code = (
                        "probe_timeout" if failure_code == "probe_failed" else "transcode_timeout"
                    )
                    raise VideoPreparationError(code, 503)
                if os.fstat(stdout.fileno()).st_size > _MAX_PROBE_BYTES:
                    raise VideoPreparationError("probe_output_limit")
                if output is not None:
                    if (
                        output.exists()
                        and output.stat().st_size > settings.video_preparation_max_output_bytes
                    ):
                        raise VideoPreparationError("output_limit", 413)
                    _disk_check(settings, reserve_output=False)
                time.sleep(0.05)
            if process.returncode != 0:
                raise VideoPreparationError(failure_code)
            if os.fstat(stdout.fileno()).st_size > _MAX_PROBE_BYTES:
                raise VideoPreparationError("probe_output_limit")
            stdout.seek(0)
            return stdout.read(_MAX_PROBE_BYTES + 1)
        finally:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.0)


def _probe(path: Path, binary: str, settings: Settings, deadline: float) -> VideoInfo:
    def read(selector: str, entries: str) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise VideoPreparationError("transcode_timeout", 503)
        raw = _run(
            [
                binary,
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-format_whitelist",
                _FORMATS,
                "-threads",
                str(settings.video_preparation_threads),
                "-select_streams",
                selector,
                "-show_entries",
                entries,
                "-of",
                "json",
                str(path),
            ],
            timeout=min(settings.video_probe_timeout_seconds, remaining),
            failure_code="probe_failed",
            settings=settings,
        )
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeError):
            raise VideoPreparationError("probe_failed") from None
        if not isinstance(payload, dict):
            raise VideoPreparationError("probe_failed")
        return payload

    payload = read(
        "v:0", "stream=codec_name,pix_fmt,width,height,duration:format=duration,format_name"
    )
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise VideoPreparationError("invalid_video")
    stream = streams[0]
    container = payload.get("format")
    if not isinstance(container, dict):
        raise VideoPreparationError("probe_failed")
    duration = _number(stream.get("duration")) or _number(container.get("duration"))
    container_duration = _number(container.get("duration"))
    if duration is None or duration <= 0.0 or container_duration is None:
        raise VideoPreparationError("duration_unknown")
    if max(duration, container_duration) > settings.video_preparation_max_duration_seconds:
        raise VideoPreparationError("duration_limit", 413)
    width, height = stream.get("width"), stream.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise VideoPreparationError("invalid_video")
    if ((width + 1) // 2 * 2) * ((height + 1) // 2 * 2) > settings.video_preparation_max_pixels:
        raise VideoPreparationError("pixel_limit", 413)
    codec, pixel_format = stream.get("codec_name"), stream.get("pix_fmt")
    if (
        not isinstance(codec, str)
        or re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", codec) is None
        or not isinstance(pixel_format, str)
        or re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", pixel_format) is None
    ):
        raise VideoPreparationError("probe_failed")
    audio = read("a:0", "stream=codec_name").get("streams", [])
    if not isinstance(audio, list) or len(audio) > 1:
        raise VideoPreparationError("probe_failed")
    audio_codec = audio[0].get("codec_name") if audio and isinstance(audio[0], dict) else None
    if audio and (
        not isinstance(audio_codec, str)
        or re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", audio_codec) is None
    ):
        raise VideoPreparationError("probe_failed")
    return VideoInfo(codec, pixel_format, width, height, duration, bool(audio), audio_codec)


def prepare_uploaded_video(path: Path, url: str, settings: Settings) -> PreparedVideo:
    """Return one verified object for both extraction and replay; keep the original intact."""
    if not settings.video_preparation_enabled:
        raise VideoPreparationError("disabled", 503)
    before = _owned_source(path, settings)
    if url != f"/data/videos/{path.name}":
        raise VideoPreparationError("invalid_source", 400)
    probe_binary = _binary(settings.video_ffprobe_binary)
    with _preparation_lock(settings) as work:
        deadline = time.monotonic() + settings.video_preparation_timeout_seconds
        digest = hashlib.sha256()
        with path.open("rb") as original:
            while block := original.read(1024 * 1024):
                digest.update(block)
                if time.monotonic() >= deadline:
                    raise VideoPreparationError("transcode_timeout", 503)
        info = _probe(path, probe_binary, settings, deadline)
        _unchanged(path, before)
        compatible = (
            path.suffix.lower() == ".mp4"
            and info.codec == "h264"
            and info.pixel_format == "yuv420p"
            and (not info.has_audio or info.audio_codec == "aac")
        )
        if compatible:
            return PreparedVideo(path, url, digest.hexdigest(), "ready_original", info)
        ffmpeg = _binary(settings.video_ffmpeg_binary)
        _disk_check(settings)
        with tempfile.TemporaryDirectory(prefix="new-upload-", dir=work) as temporary:
            output = Path(temporary) / "playback.partial"
            arguments = [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-protocol_whitelist",
                "file",
                "-format_whitelist",
                _FORMATS,
                "-filter_threads",
                str(settings.video_preparation_threads),
                "-threads",
                str(settings.video_preparation_threads),
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-sn",
                "-dn",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                str(settings.video_preparation_threads),
                "-fps_mode",
                "passthrough",
                "-vf",
                "setpts=PTS-STARTPTS,pad=ceil(iw/2)*2:ceil(ih/2)*2",
            ]
            if settings.video_preparation_include_audio and info.has_audio:
                arguments.extend(
                    [
                        "-map",
                        "0:a:0",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "96k",
                        "-ac",
                        "2",
                        "-af",
                        "asetpts=PTS-STARTPTS",
                    ]
                )
            else:
                arguments.append("-an")
            arguments.extend(
                [
                    "-movflags",
                    "+faststart",
                    "-fs",
                    str(settings.video_preparation_max_output_bytes),
                    "-f",
                    "mp4",
                    str(output),
                ]
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise VideoPreparationError("transcode_timeout", 503)
            _run(
                arguments,
                timeout=remaining,
                failure_code="transcode_failed",
                settings=settings,
                output=output,
            )
            if (
                not output.is_file()
                or output.is_symlink()
                or output.stat().st_nlink != 1
                or output.stat().st_size == 0
            ):
                raise VideoPreparationError("transcode_failed")
            if output.stat().st_size >= settings.video_preparation_max_output_bytes:
                raise VideoPreparationError("output_limit", 413)
            prepared = _probe(output, probe_binary, settings, deadline)
            if prepared.codec != "h264" or prepared.pixel_format != "yuv420p":
                raise VideoPreparationError("output_not_compatible")
            expected_audio = settings.video_preparation_include_audio and info.has_audio
            if prepared.has_audio != expected_audio or (
                expected_audio and prepared.audio_codec != "aac"
            ):
                raise VideoPreparationError("output_not_compatible")
            if abs(prepared.duration_seconds - info.duration_seconds) > 0.1:
                raise VideoPreparationError("output_incomplete")
            _unchanged(path, before)
            if settings.videos_dir.is_symlink():
                raise VideoPreparationError("invalid_source", 400)
            target = settings.videos_dir / f"{uuid.uuid4()}-playback.mp4"
            os.chmod(output, 0o600)
            with output.open("rb") as completed:
                os.fsync(completed.fileno())
            # Non-overwriting atomic publication; source and target are on the same filesystem.
            os.link(output, target)
            output.unlink()
            return PreparedVideo(
                target,
                f"/data/videos/{target.name}",
                digest.hexdigest(),
                "ready_normalized",
                prepared,
            )
