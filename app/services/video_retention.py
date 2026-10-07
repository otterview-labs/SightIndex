"""Retention guards for evidence owned by uploaded video assets.

Existing image cleanup tasks are not video-retention policies. Preserve videos
regardless of their processing state; deleting them needs separate authorization.
"""

from __future__ import annotations

import logging
import re
import stat
from collections.abc import Iterable
from pathlib import Path

from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from app.models.media import VideoAsset
from app.services.storage import IMAGE_SUFFIXES

logger = logging.getLogger(__name__)
_IMAGE_URL = re.compile(r"/data/(frames|uploads|crops|thumbnails)/[A-Za-z0-9][A-Za-z0-9._-]*")


def video_asset_media_urls(db: Session) -> frozenset[str]:
    """Protect originals and derivatives, including failed or interrupted assets.

    Legacy databases without this additive table are still safe: the deletion
    helper independently forbids all video paths, not only indexed video URLs.
    Other database errors deliberately propagate before any cleanup mutations.
    """
    if not inspect(db.get_bind()).has_table(VideoAsset.__tablename__):
        return frozenset()
    return frozenset(
        url
        for row in db.execute(select(VideoAsset.original_video_url, VideoAsset.playback_video_url))
        for url in row
        if url
    )


def remove_retired_image_files(
    data_dir: Path,
    urls: Iterable[tuple[str | None, str | None]],
    protected_urls: frozenset[str] = frozenset(),
) -> int:
    """Remove only internal single-file image URLs; never follow links or videos."""
    root = data_dir.resolve()
    removed = 0
    for row in urls:
        for url in row:
            if (
                not isinstance(url, str)
                or url in protected_urls
                or _IMAGE_URL.fullmatch(url) is None
                or Path(url).suffix.lower() not in IMAGE_SUFFIXES
            ):
                continue
            target = root / url.removeprefix("/data/")
            try:
                if target.parent.is_symlink() or target.is_symlink():
                    continue
                if not target.resolve().is_relative_to(root):
                    continue
                identity = target.stat()
                if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
                    continue
                target.unlink()
                removed += 1
            except FileNotFoundError:
                continue
            except (OSError, RuntimeError):
                # Do not echo arbitrary supplied paths or OS diagnostics.
                logger.warning("Could not remove one retired image file")
    return removed
