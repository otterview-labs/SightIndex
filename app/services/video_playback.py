"""Resolve stored frame provenance without inventing history from live camera URLs."""

import math
import re
import stat
from pathlib import Path

from app.config.settings import Settings
from app.models.media import Image, VideoAsset
from app.schemas.media import VideoPlaybackRead
from app.services.storage import VIDEO_SUFFIXES

_VIDEO_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def video_playback_for_image(
    image: Image, settings: Settings, asset: VideoAsset | None = None
) -> VideoPlaybackRead:
    """Return only local, regular source media with a finite, known frame position.

    This does not contact a camera, inspect a video, or infer legacy provenance. A
    zero-second location is valid; a missing location is never converted to zero.
    """

    def unavailable(reason: str) -> VideoPlaybackRead:
        return VideoPlaybackRead(
            available=False,
            source_type=image.source_type,
            captured_at=image.captured_at,
            reason=reason,
        )

    if image.source_type in {"stream", "stream_frame", "stream_snapshot"}:
        return unavailable("recording_not_configured")
    if image.source_type != "video_frame":
        return unavailable("not_video")
    video_url = image.source_video_url
    if not video_url:
        return unavailable("source_missing")

    prefix = "/data/videos/"
    filename = video_url.removeprefix(prefix)
    if (
        not video_url.startswith(prefix)
        or _VIDEO_FILENAME.fullmatch(filename) is None
        or Path(filename).suffix.lower() not in VIDEO_SUFFIXES
    ):
        return unavailable("invalid_source")
    videos_dir = settings.videos_dir
    source_path = videos_dir / filename
    try:
        # Reject links even when their target is inside the configured media directory.
        if videos_dir.is_symlink() or source_path.is_symlink():
            return unavailable("invalid_source")
        if not source_path.resolve().is_relative_to(videos_dir.resolve()):
            return unavailable("invalid_source")
        if not stat.S_ISREG(source_path.stat().st_mode):
            return unavailable("invalid_source")
    except (OSError, RuntimeError):
        return unavailable("media_missing")

    raw_offset = image.video_offset_seconds
    if raw_offset is None or isinstance(raw_offset, bool):
        return unavailable("offset_unknown")
    try:
        offset = float(raw_offset)
    except (TypeError, ValueError, OverflowError):
        return unavailable("offset_unknown")
    if not math.isfinite(offset) or offset < 0:
        return unavailable("offset_unknown")
    if asset is not None:
        if asset.playback_video_url != video_url:
            return unavailable("invalid_source")
        if asset.preparation_status not in {"ready_original", "ready_normalized"}:
            return unavailable("video_not_ready")
        duration = asset.duration_seconds
        if (
            duration is None
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration <= 0.0
            or offset >= duration
        ):
            return unavailable("offset_out_of_range")
    original_url = asset.original_video_url if asset is not None else None
    if original_url is not None:
        name = original_url.removeprefix(prefix)
        if (
            not original_url.startswith(prefix)
            or _VIDEO_FILENAME.fullmatch(name) is None
            or Path(name).suffix.lower() not in VIDEO_SUFFIXES
        ):
            original_url = None
    return VideoPlaybackRead(
        available=True,
        source_type=image.source_type,
        video_url=video_url,
        offset_seconds=float(offset),
        captured_at=image.captured_at,
        video_id=asset.id if asset is not None else None,
        original_video_url=original_url,
        compatibility_status=asset.preparation_status if asset is not None else "unknown",
        has_audio=asset.has_audio if asset is not None else None,
    )
