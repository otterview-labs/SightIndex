import logging
import math
import time
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.models.events import CountingEvent, RecognitionEvent
from app.models.media import Image, PersonCrop, VideoAsset
from app.schemas.media import VideoProcessResponse
from app.services.frame_processing import Detection, FrameProcessingService
from app.services.media import MediaService
from app.services.storage import StorageService
from app.services.time_utils import database_datetime, local_now
from app.services.vector_index_queue import VectorQueueFullError
from app.services.video_preparation import VERSION, VideoPreparationError, prepare_uploaded_video

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoFrameFile:
    url: str
    path: Path
    captured_at: datetime


@dataclass(frozen=True)
class CountingLine:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class PersonTrack:
    id: int
    center: tuple[float, float]
    side: float
    counted: bool = False
    last_seen_at: float | None = None
    missed_frames: int = 0


@dataclass(frozen=True)
class LineCrossing:
    detection_index: int
    direction: str


class VideoProcessingBackpressureError(VectorQueueFullError):
    """Carries the exact committed progress when queue pressure stops an upload."""

    def __init__(
        self,
        cause: VectorQueueFullError,
        *,
        frames_read: int,
        frames_sampled: int,
        frames_processed: int,
        image_ids: list[uuid.UUID],
        crop_ids: list[uuid.UUID],
        counting_events_created: int,
        video_id: uuid.UUID | None = None,
    ) -> None:
        self.cause = cause
        self.frames_read = frames_read
        self.frames_sampled = frames_sampled
        self.frames_processed = frames_processed
        self.image_ids = tuple(image_ids)
        self.crop_ids = tuple(crop_ids)
        self.counting_events_created = counting_events_created
        self.video_id = video_id
        super().__init__(
            "vector index queue is full; "
            f"video processing stopped after {frames_processed} completed sampled frames"
        )

    def api_detail(self) -> dict[str, object]:
        detail: dict[str, object] = {
            "code": "vector_index_queue_full",
            "message": str(self),
            "cause": str(self.cause),
            "partial": bool(
                self.frames_processed
                or self.image_ids
                or self.crop_ids
                or self.counting_events_created
            ),
            "frames_read": self.frames_read,
            "frames_sampled": self.frames_sampled,
            "frames_processed": self.frames_processed,
            "images_committed": len(self.image_ids),
            "crops_committed": len(self.crop_ids),
            "counting_events_committed": self.counting_events_created,
            "image_ids": [str(image_id) for image_id in self.image_ids],
            "crop_ids": [str(crop_id) for crop_id in self.crop_ids],
        }
        if self.video_id is not None:
            detail["video_id"] = str(self.video_id)
        return detail


class VideoProcessingService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.storage = StorageService(settings)
        self.media = MediaService(db, settings)
        self.processor = FrameProcessingService(db, settings)

    def process_upload(
        self,
        file: UploadFile,
        frame_interval_seconds: float = 1.0,
        max_frames: int = 120,
        store_empty_frames: bool | None = None,
        counting_line: CountingLine | None = None,
        camera_id: uuid.UUID | None = None,
        location_id: uuid.UUID | None = None,
        captured_at: datetime | None = None,
    ) -> VideoProcessResponse:
        video_url = self.storage.save_video(file)
        video_path = self._resolve_data_url(video_url)
        if video_path is None:
            self.storage.remove_data_url(video_url)
            raise ValueError("Uploaded video path cannot be resolved")
        if self.settings.video_preparation_enabled:
            return self._process_prepared_upload(
                video_path=video_path,
                original_url=video_url,
                frame_interval_seconds=frame_interval_seconds,
                max_frames=max_frames,
                store_empty_frames=store_empty_frames,
                counting_line=counting_line,
                camera_id=camera_id,
                location_id=location_id,
                captured_at=captured_at,
            )
        try:
            return self.process_video_path(
                video_path=video_path,
                video_url=video_url,
                frame_interval_seconds=frame_interval_seconds,
                max_frames=max_frames,
                store_empty_frames=store_empty_frames,
                counting_line=counting_line,
                camera_id=camera_id,
                location_id=location_id,
                captured_at=captured_at,
            )
        except VideoProcessingBackpressureError as exc:
            # Completed frames already reference this source; preserve their replay evidence.
            if not exc.image_ids:
                self.storage.remove_data_url(video_url)
            raise
        except Exception:
            try:
                self.db.rollback()
                has_committed_frames = (
                    self.db.scalar(
                        select(Image.id).where(Image.source_video_url == video_url).limit(1)
                    )
                    is not None
                )
            except Exception:
                # A database outage must not remove a possibly committed source video.
                logger.warning("Could not verify uploaded video ownership; preserving source")
            else:
                if not has_committed_frames:
                    self.storage.remove_data_url(video_url)
            raise

    def _process_prepared_upload(
        self,
        *,
        video_path: Path,
        original_url: str,
        frame_interval_seconds: float,
        max_frames: int,
        store_empty_frames: bool | None,
        counting_line: CountingLine | None,
        camera_id: uuid.UUID | None,
        location_id: uuid.UUID | None,
        captured_at: datetime | None,
    ) -> VideoProcessResponse:
        """Persist ownership before preparation; never overwrite or delete the original."""
        asset = VideoAsset(
            original_video_url=original_url,
            preparation_status="preparing",
            processing_status="pending",
            preparation_version=VERSION,
        )
        self.db.add(asset)
        self.db.commit()
        self.db.refresh(asset)
        asset_id = asset.id
        # Release the refresh transaction before a potentially slow external CPU process.
        self.db.commit()
        try:
            prepared = prepare_uploaded_video(video_path, original_url, self.settings)
        except VideoPreparationError as exc:
            exc.video_id = asset_id
            self._record_video_asset_status(asset_id, "failed", "pending", exc.code)
            raise
        except Exception:
            self._record_video_asset_status(asset_id, "failed", "pending", "preparation_failed")
            failure = VideoPreparationError("preparation_failed", 503)
            failure.video_id = asset_id
            raise failure from None

        asset = self.db.get(VideoAsset, asset_id)
        if asset is None:
            raise VideoPreparationError("asset_missing", 503)
        asset.original_sha256 = prepared.original_sha256
        asset.playback_video_url = prepared.url
        asset.preparation_status = prepared.status
        asset.processing_status = "processing"
        asset.codec = prepared.info.codec
        asset.duration_seconds = prepared.info.duration_seconds
        asset.has_audio = prepared.info.has_audio
        self.db.commit()
        try:
            result = self.process_video_path(
                video_path=prepared.path,
                video_url=prepared.url,
                frame_interval_seconds=frame_interval_seconds,
                max_frames=max_frames,
                store_empty_frames=store_empty_frames,
                counting_line=counting_line,
                camera_id=camera_id,
                location_id=location_id,
                captured_at=captured_at,
            )
        except VideoProcessingBackpressureError as exc:
            exc.video_id = asset_id
            self._record_video_asset_status(
                asset_id,
                prepared.status,
                "partial" if exc.image_ids else "failed",
                "vector_index_queue_full",
            )
            raise
        except Exception:
            self._record_video_asset_status(
                asset_id, prepared.status, "failed", "processing_failed"
            )
            failure = VideoPreparationError("processing_failed", 422)
            failure.video_id = asset_id
            raise failure from None
        self._record_video_asset_status(asset_id, prepared.status, "complete", None)
        return result.model_copy(
            update={
                "video_id": asset_id,
                "original_video_url": original_url,
                "compatibility_status": prepared.status,
                "has_audio": prepared.info.has_audio,
            }
        )

    def _record_video_asset_status(
        self, asset_id: uuid.UUID, preparation: str, processing: str, reason: str | None
    ) -> None:
        """Best-effort metadata update without masking a failure or deleting media."""
        try:
            self.db.rollback()
            asset = self.db.get(VideoAsset, asset_id)
            if asset is not None:
                asset.preparation_status = preparation
                asset.processing_status = processing
                asset.reason = reason
                self.db.commit()
        except Exception:
            with suppress(Exception):
                self.db.rollback()
            logger.warning("Could not update prepared video ownership state")

    def process_video_path(
        self,
        video_path: Path,
        video_url: str,
        frame_interval_seconds: float = 1.0,
        max_frames: int = 120,
        store_empty_frames: bool | None = None,
        counting_line: CountingLine | None = None,
        camera_id: uuid.UUID | None = None,
        location_id: uuid.UUID | None = None,
        captured_at: datetime | None = None,
    ) -> VideoProcessResponse:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(f"OpenCV is not installed: {exc}") from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Could not open uploaded video")

        should_store_empty = (
            self.settings.stream_store_empty_frames
            if store_empty_frames is None
            else store_empty_frames
        )
        fps = self._finite_nonnegative(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        sample_step = fps * frame_interval_seconds
        frame_step = max(1, round(sample_step)) if math.isfinite(sample_step) else 1
        base_captured_at = database_datetime(
            captured_at or local_now(self.settings),
            self.settings,
            self.db.get_bind().dialect.name,
        )

        frames_read = 0
        frames_sampled = 0
        frames_processed = 0
        image_ids: list[uuid.UUID] = []
        crop_ids: list[uuid.UUID] = []
        counting_events_created = 0
        tracks: dict[int, PersonTrack] = {}
        next_track_id = 1
        active_frame_path: Path | None = None
        previous_media_position_ms: float | None = None

        try:
            while frames_sampled < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index = frames_read
                frames_read += 1
                # Inspect every decoded frame, including frames omitted by sampling. A clock
                # reversal on an omitted frame must not make the next stored frame look valid.
                position_ms = self._finite_nonnegative(capture.get(cv2.CAP_PROP_POS_MSEC))
                video_offset_seconds = self._offset_from_media_position(
                    position_ms,
                    frame_index=frame_index,
                    fps=fps,
                    previous_position_ms=previous_media_position_ms,
                )
                if position_ms is not None:
                    # Keep the high-water mark: a broken clock must recover past its earlier
                    # value before we trust it again. Never guess a seek position from FPS.
                    previous_media_position_ms = max(previous_media_position_ms or 0.0, position_ms)
                if frame_index % frame_step != 0:
                    continue

                frames_sampled += 1
                frame_file = self._write_frame_file(
                    video_path=video_path,
                    frame=frame,
                    cv2=cv2,
                    captured_at=self._captured_at_for_offset(
                        base_captured_at, video_offset_seconds
                    ),
                )
                active_frame_path = frame_file.path
                frame_height, frame_width = frame.shape[:2]
                detections = self.processor.quality_filter_detections(
                    self.processor.detect_image_path(frame_file.path), frame_width, frame_height
                )
                if counting_line is not None:
                    crossings, next_track_id = self._line_crossings(
                        detections=detections,
                        frame=frame,
                        line=counting_line,
                        tracks=tracks,
                        next_track_id=next_track_id,
                        observed_at=frame_file.captured_at.timestamp(),
                        max_idle_seconds=max(
                            self.settings.line_crossing_track_idle_seconds,
                            frame_interval_seconds * 2.0,
                        ),
                    )
                    if not crossings:
                        frame_file.path.unlink(missing_ok=True)
                        active_frame_path = None
                        frames_processed += 1
                        continue

                    image = self._create_frame_image(
                        frame_file,
                        camera_id=camera_id,
                        location_id=location_id,
                        source_video_url=video_url,
                        video_offset_seconds=video_offset_seconds,
                    )
                    crossing_detections = self._crossing_detections(detections, crossings)
                    crops = self.processor.process_image(image, detections=crossing_detections)
                    self._try_index_frame_image(image)
                    crops_by_detection_index = self._crops_by_detection_index(crops)
                    created = self._create_line_crossing_events(
                        crossings=crossings,
                        crops_by_detection_index=crops_by_detection_index,
                        counted_at=frame_file.captured_at,
                        camera_id=camera_id,
                        location_id=location_id,
                        image=image,
                    )
                    self.db.commit()
                    counting_events_created += created
                    image_ids.append(image.id)
                    crop_ids.extend(crop.id for crop in crops)
                elif detections or should_store_empty:
                    image = self._create_frame_image(
                        frame_file,
                        camera_id=camera_id,
                        location_id=location_id,
                        source_video_url=video_url,
                        video_offset_seconds=video_offset_seconds,
                    )
                    crops = self.processor.process_image(image, detections=detections)
                    self._try_index_frame_image(image)
                    image_ids.append(image.id)
                    crop_ids.extend(crop.id for crop in crops)
                else:
                    frame_file.path.unlink(missing_ok=True)
                active_frame_path = None
                frames_processed += 1
        except VectorQueueFullError as exc:
            self.db.rollback()
            if active_frame_path is not None:
                active_frame_path.unlink(missing_ok=True)
            raise VideoProcessingBackpressureError(
                exc,
                frames_read=frames_read,
                frames_sampled=frames_sampled,
                frames_processed=frames_processed,
                image_ids=image_ids,
                crop_ids=crop_ids,
                counting_events_created=counting_events_created,
            ) from exc
        finally:
            capture.release()

        self.db.commit()
        return VideoProcessResponse(
            video_url=video_url,
            frame_interval_seconds=frame_interval_seconds,
            frames_read=frames_read,
            frames_sampled=frames_sampled,
            images_created=len(image_ids),
            crops_created=len(crop_ids),
            counting_events_created=counting_events_created,
            image_ids=image_ids,
            crop_ids=crop_ids,
        )

    @staticmethod
    def create_counting_line(
        x1: float | None,
        y1: float | None,
        x2: float | None,
        y2: float | None,
    ) -> CountingLine | None:
        values = (x1, y1, x2, y2)
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError("Counting line requires line_x1, line_y1, line_x2 and line_y2")
        line = CountingLine(x1=x1, y1=y1, x2=x2, y2=y2)  # type: ignore[arg-type]
        if (line.x1, line.y1) == (line.x2, line.y2):
            raise ValueError("Counting line must have two different points")
        return line

    def _create_frame_image(
        self,
        frame_file: VideoFrameFile,
        camera_id: uuid.UUID | None = None,
        location_id: uuid.UUID | None = None,
        source_video_url: str | None = None,
        video_offset_seconds: float | None = None,
    ) -> Image:
        image = Image(
            image_url=frame_file.url,
            source_type="video_frame",
            source_video_url=source_video_url,
            video_offset_seconds=video_offset_seconds,
            camera_id=camera_id,
            location_id=location_id,
            captured_at=frame_file.captured_at,
        )
        self.db.add(image)
        self.db.flush()
        return image

    def _try_index_frame_image(self, image: Image) -> None:
        if self.settings.vector_index_on_ingest:
            self.media._try_index_image(image)

    def _write_frame_file(
        self,
        video_path: Path,
        frame: Any,
        cv2: Any,
        captured_at: datetime,
    ) -> VideoFrameFile:
        self.settings.frames_dir.mkdir(parents=True, exist_ok=True)
        # Duplicate/unknown decoder timestamps must not overwrite an earlier frame's pixels.
        filename = (
            f"{video_path.stem}_{captured_at.strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4().hex}.jpg"
        )
        path = self.settings.frames_dir / filename
        if not cv2.imwrite(
            str(path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.settings.frame_jpeg_quality)],
        ):
            raise RuntimeError("Could not write extracted video frame")
        return VideoFrameFile(url=f"/data/frames/{filename}", path=path, captured_at=captured_at)

    def _frame_captured_at(self, capture: Any, cv2: Any, base_captured_at: datetime) -> datetime:
        """Compatibility helper for direct callers; unknown times retain the base stamp."""

        offset = self._frame_offset_seconds(capture, cv2, frame_index=0, fps=0.0)
        return self._captured_at_for_offset(base_captured_at, offset)

    @staticmethod
    def _finite_nonnegative(value: object) -> float | None:
        """Normalize decoder numbers without allowing NaN, infinity, or negative times."""

        if isinstance(value, bool):
            return None
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    @classmethod
    def _frame_offset_seconds(
        cls,
        capture: Any,
        cv2: Any,
        *,
        frame_index: int,
        fps: float,
        previous_position_ms: float | None = None,
    ) -> float | None:
        """Read this frame's media clock; FPS alone cannot establish a reliable seek time.

        The FPS argument remains accepted for compatibility, but is never used to infer a
        playback offset. It may be an average for variable-rate media, and does not establish
        a fixed frame rate or the stream's starting timestamp.
        """

        position_ms = cls._finite_nonnegative(capture.get(cv2.CAP_PROP_POS_MSEC))
        return cls._offset_from_media_position(
            position_ms,
            frame_index=frame_index,
            fps=fps,
            previous_position_ms=previous_position_ms,
        )

    @classmethod
    def _offset_from_media_position(
        cls,
        position_ms: float | None,
        *,
        frame_index: int,
        fps: float,
        previous_position_ms: float | None = None,
    ) -> float | None:
        """Preserve increasing decoder times and abstain for absent or broken media clocks.

        ``previous_position_ms`` is the high-water mark of all earlier decoded frames, not
        merely the previous stored frame. A recovered clock must exceed that mark. Zero is
        legitimate for the first decoded frame, but not for later unsupported/stalled clocks.
        ``fps`` is retained for existing callers and intentionally does not provide a fallback.
        """

        position_ms = cls._finite_nonnegative(position_ms)
        if (
            frame_index >= 0
            and position_ms is not None
            and (position_ms > 0 or frame_index == 0)
            and (previous_position_ms is None or position_ms > previous_position_ms)
        ):
            return position_ms / 1000.0
        return None

    @staticmethod
    def _captured_at_for_offset(
        base_captured_at: datetime, offset_seconds: float | None
    ) -> datetime:
        """Keep display timestamps usable when the decoder has no trustworthy media clock."""

        if offset_seconds is None:
            return base_captured_at
        try:
            return base_captured_at + timedelta(seconds=offset_seconds)
        except OverflowError:
            return base_captured_at

    def _count_line_crossings(
        self,
        detections: list[Detection],
        frame: Any,
        line: CountingLine,
        tracks: dict[int, PersonTrack],
        next_track_id: int,
        counted_at: datetime,
        camera_id: uuid.UUID | None,
        location_id: uuid.UUID | None,
        stream_id: uuid.UUID | None = None,
        image: Image | None = None,
        crops: list[PersonCrop] | None = None,
        now: float | None = None,
        max_idle_seconds: float | None = None,
    ) -> tuple[int, int]:
        crossings, next_track_id = self._line_crossings(
            detections=detections,
            frame=frame,
            line=line,
            tracks=tracks,
            next_track_id=next_track_id,
            observed_at=counted_at.timestamp() if now is None else now,
            max_idle_seconds=max_idle_seconds,
        )
        crop_items = crops or []
        crops_by_detection_index = self._crops_by_detection_index(crop_items)
        has_explicit_association = any("source_detection_index" in crop.bbox for crop in crop_items)
        if not has_explicit_association and len(crop_items) == len(detections):
            # Compatibility for old direct callers that provide one crop for every detection.
            # A partial list without explicit source indices must abstain instead of shifting
            # crop identities onto the wrong crossing.
            crops_by_detection_index = dict(enumerate(crop_items))
        count = self._create_line_crossing_events(
            crossings=crossings,
            crops_by_detection_index=crops_by_detection_index,
            counted_at=counted_at,
            camera_id=camera_id,
            location_id=location_id,
            stream_id=stream_id,
            image=image,
        )
        return count, next_track_id

    @staticmethod
    def _crossing_detections(
        detections: list[Detection],
        crossings: list[LineCrossing],
    ) -> list[Detection]:
        """Attach the original input index without trusting mutable bbox payloads."""

        return [
            replace(
                detections[crossing.detection_index],
                source_index=crossing.detection_index,
            )
            for crossing in crossings
        ]

    @staticmethod
    def _crops_by_detection_index(
        crops: list[PersonCrop],
    ) -> dict[int, PersonCrop]:
        """Map only crops that retain their successful source detection association."""

        result: dict[int, PersonCrop] = {}
        for crop in crops:
            source_index = crop.bbox.get("source_detection_index")
            if isinstance(source_index, int) and not isinstance(source_index, bool):
                result[source_index] = crop
        return result

    def _create_line_crossing_events(
        self,
        crossings: list[LineCrossing],
        crops_by_detection_index: Mapping[int, PersonCrop],
        counted_at: datetime,
        camera_id: uuid.UUID | None,
        location_id: uuid.UUID | None,
        stream_id: uuid.UUID | None = None,
        image: Image | None = None,
    ) -> int:
        for crossing in crossings:
            crop = crops_by_detection_index.get(crossing.detection_index)
            recognition_event = self._recognition_event_for_crop(crop)
            if (
                crop is not None
                and crop.identity_is_protected
                and recognition_event is not None
                and recognition_event.person_id != crop.person_id
            ):
                recognition_event = None
            person_id = (
                recognition_event.person_id
                if recognition_event and recognition_event.person_id
                else crop.person_id
                if crop
                else None
            )
            self.db.add(
                CountingEvent(
                    stream_id=stream_id,
                    image_id=image.id if image else None,
                    crop_id=crop.id if crop else None,
                    recognition_event_id=recognition_event.id if recognition_event else None,
                    person_id=person_id,
                    unknown_cluster_id=(
                        recognition_event.unknown_cluster_id if recognition_event else None
                    ),
                    camera_id=(crop.camera_id if crop else camera_id),
                    location_id=(crop.location_id if crop else location_id),
                    count_type="line_crossing",
                    direction=crossing.direction,
                    counted_at=counted_at,
                )
            )
        return len(crossings)

    def _line_crossings(
        self,
        detections: list[Detection],
        frame: Any,
        line: CountingLine,
        tracks: dict[int, PersonTrack],
        next_track_id: int,
        now: float | None = None,
        max_idle_seconds: float | None = None,
        *,
        observed_at: float | None = None,
    ) -> tuple[list[LineCrossing], int]:
        current_time = self._expire_tracks(
            tracks,
            now=observed_at if observed_at is not None else now,
            max_idle_seconds=max_idle_seconds,
        )
        height, width = frame.shape[:2]
        matched_track_ids: set[int] = set()
        crossings: list[LineCrossing] = []
        for detection_index, detection in enumerate(detections):
            center = self._detection_center(detection, width, height)
            side = self._line_side(line, center)
            if not all(math.isfinite(value) for value in (*center, side)):
                continue
            track = self._match_track(center, tracks, matched_track_ids)
            if track is None:
                tracks[next_track_id] = PersonTrack(
                    id=next_track_id,
                    center=center,
                    side=side,
                    last_seen_at=current_time,
                )
                matched_track_ids.add(next_track_id)
                next_track_id += 1
                continue

            matched_track_ids.add(track.id)
            track.last_seen_at = current_time
            track.missed_frames = 0
            if not track.counted and self._crossed_line(track.side, side):
                direction = "a_to_b" if track.side < side else "b_to_a"
                crossings.append(LineCrossing(detection_index=detection_index, direction=direction))
                track.counted = True
            track.center = center
            if abs(side) > 0.0001:
                track.side = side
        for track_id, track in list(tracks.items()):
            if track_id not in matched_track_ids:
                track.missed_frames += 1
                if track.missed_frames >= getattr(
                    self.settings, "line_crossing_track_max_missed_frames", 2
                ):
                    del tracks[track_id]
        return crossings, next_track_id

    def _expire_tracks(
        self,
        tracks: dict[int, PersonTrack],
        *,
        now: float | None,
        max_idle_seconds: float | None = None,
    ) -> float:
        """Drop stale or corrupt tracks and return the timestamp used for this frame.

        Explicit timestamps let uploaded videos advance by media time while live callers use a
        monotonic clock.  A missing or invalid explicit timestamp safely falls back to monotonic
        time.  If a source clock moves backwards, existing tracks are rebased instead of gaining
        an unbounded negative age.
        """

        current_time = float(now) if now is not None else time.monotonic()
        if not math.isfinite(current_time):
            current_time = time.monotonic()
        configured_idle = (
            max_idle_seconds
            if max_idle_seconds is not None
            else getattr(
                self.settings,
                "line_crossing_track_idle_seconds",
                getattr(self.settings, "counting_track_idle_seconds", 6.0),
            )
        )
        idle_seconds = float(configured_idle)
        if not math.isfinite(idle_seconds) or idle_seconds <= 0:
            idle_seconds = 6.0

        for track_id, track in list(tracks.items()):
            state = (*track.center, track.side)
            if not all(math.isfinite(value) for value in state):
                del tracks[track_id]
                continue
            if track.last_seen_at is None:
                # Tracks constructed by older callers get one normal idle window to match.
                track.last_seen_at = current_time
                continue
            if not math.isfinite(track.last_seen_at):
                del tracks[track_id]
                continue
            age = current_time - track.last_seen_at
            if age < 0:
                track.last_seen_at = current_time
            elif age > idle_seconds or (max_idle_seconds is None and age == idle_seconds):
                # The configured idle limit expires at its boundary. An explicit sampling
                # window includes its last observation, keeping exactly two intervals usable.
                del tracks[track_id]
        return current_time

    def _recognition_event_for_crop(self, crop: PersonCrop | None) -> RecognitionEvent | None:
        if crop is None:
            return None
        return self.db.scalar(
            select(RecognitionEvent)
            .where(RecognitionEvent.crop_id == crop.id)
            .order_by(RecognitionEvent.created_at.desc())
            .limit(1)
        )

    def _detection_center(
        self,
        detection: Detection,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float]:
        bbox = detection.bbox
        x = float(bbox.get("x", 0))
        y = float(bbox.get("y", 0))
        width = float(bbox.get("width", 0))
        height = float(bbox.get("height", 0))
        center_x = (x + width / 2) / frame_width
        if self.settings.line_crossing_point == "center":
            center_y = (y + height / 2) / frame_height
        else:
            center_y = (y + height) / frame_height
        if not all(math.isfinite(value) for value in (center_x, center_y)):
            return center_x, center_y
        return (
            max(0.0, min(1.0, center_x)),
            max(0.0, min(1.0, center_y)),
        )

    def _line_side(self, line: CountingLine, point: tuple[float, float]) -> float:
        return (line.x2 - line.x1) * (point[1] - line.y1) - (line.y2 - line.y1) * (
            point[0] - line.x1
        )

    def _crossed_line(self, previous_side: float, current_side: float) -> bool:
        if abs(previous_side) <= 0.0001 or abs(current_side) <= 0.0001:
            return False
        return previous_side * current_side < 0

    def _match_track(
        self,
        center: tuple[float, float],
        tracks: dict[int, PersonTrack],
        matched_track_ids: set[int],
    ) -> PersonTrack | None:
        best_track: PersonTrack | None = None
        best_distance = self.settings.line_crossing_match_distance
        for track in tracks.values():
            if track.id in matched_track_ids:
                continue
            distance = (
                (track.center[0] - center[0]) ** 2 + (track.center[1] - center[1]) ** 2
            ) ** 0.5
            if distance < best_distance:
                best_distance = distance
                best_track = track
        return best_track

    def _resolve_data_url(self, url: str) -> Path | None:
        prefix = "/data/"
        if not url.startswith(prefix):
            return None
        return self.settings.data_dir / url.removeprefix(prefix)
