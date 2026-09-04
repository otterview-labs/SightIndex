import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.models.events import CountingEvent, RecognitionEvent
from app.models.media import Image, PersonCrop
from app.schemas.media import VideoProcessResponse
from app.services.frame_processing import Detection, FrameProcessingService
from app.services.media import MediaService
from app.services.storage import StorageService
from app.services.time_utils import database_datetime, local_now
from app.services.vector_index_queue import VectorQueueFullError


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
    ) -> None:
        self.cause = cause
        self.frames_read = frames_read
        self.frames_sampled = frames_sampled
        self.frames_processed = frames_processed
        self.image_ids = tuple(image_ids)
        self.crop_ids = tuple(crop_ids)
        self.counting_events_created = counting_events_created
        super().__init__(
            "vector index queue is full; "
            f"video processing stopped after {frames_processed} completed sampled frames"
        )

    def api_detail(self) -> dict[str, object]:
        return {
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
        except Exception:
            self.storage.remove_data_url(video_url)
            raise

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
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_step = max(1, round(fps * frame_interval_seconds)) if fps > 0 else 1
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

        try:
            while frames_sampled < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index = frames_read
                frames_read += 1
                if frame_index % frame_step != 0:
                    continue

                frames_sampled += 1
                frame_file = self._write_frame_file(
                    video_path=video_path,
                    frame=frame,
                    cv2=cv2,
                    captured_at=self._frame_captured_at(capture, cv2, base_captured_at),
                )
                active_frame_path = frame_file.path
                detections = self.processor.quality_filter_detections(
                    self.processor.detect_image_path(frame_file.path)
                )
                if counting_line is not None:
                    crossings: list[LineCrossing] = []
                    if detections:
                        crossings, next_track_id = self._line_crossings(
                            detections=detections,
                            frame=frame,
                            line=counting_line,
                            tracks=tracks,
                            next_track_id=next_track_id,
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
                    )
                    crossing_detections = [
                        detections[crossing.detection_index] for crossing in crossings
                    ]
                    crops = self.processor.process_image(image, detections=crossing_detections)
                    self._try_index_frame_image(image)
                    crops_by_detection_index = {
                        crossing.detection_index: crop
                        for crossing, crop in zip(crossings, crops, strict=False)
                    }
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
    ) -> Image:
        image = Image(
            image_url=frame_file.url,
            source_type="video_frame",
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
        filename = f"{video_path.stem}_{captured_at.strftime('%Y%m%d%H%M%S%f')}.jpg"
        path = self.settings.frames_dir / filename
        if not cv2.imwrite(
            str(path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.settings.frame_jpeg_quality)],
        ):
            raise RuntimeError("Could not write extracted video frame")
        return VideoFrameFile(url=f"/data/frames/{filename}", path=path, captured_at=captured_at)

    def _frame_captured_at(self, capture: Any, cv2: Any, base_captured_at: datetime) -> datetime:
        position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        return base_captured_at + timedelta(milliseconds=position_ms)

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
    ) -> tuple[int, int]:
        crossings, next_track_id = self._line_crossings(
            detections=detections,
            frame=frame,
            line=line,
            tracks=tracks,
            next_track_id=next_track_id,
        )
        crops_by_detection_index = {
            index: crop for index, crop in enumerate(crops or [])
        }
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
    ) -> tuple[list[LineCrossing], int]:
        height, width = frame.shape[:2]
        matched_track_ids: set[int] = set()
        crossings: list[LineCrossing] = []
        for detection_index, detection in enumerate(detections):
            center = self._detection_center(detection, width, height)
            side = self._line_side(line, center)
            track = self._match_track(center, tracks, matched_track_ids)
            if track is None:
                tracks[next_track_id] = PersonTrack(id=next_track_id, center=center, side=side)
                matched_track_ids.add(next_track_id)
                next_track_id += 1
                continue

            matched_track_ids.add(track.id)
            if not track.counted and self._crossed_line(track.side, side):
                direction = "a_to_b" if track.side < side else "b_to_a"
                crossings.append(LineCrossing(detection_index=detection_index, direction=direction))
                track.counted = True
            track.center = center
            if abs(side) > 0.0001:
                track.side = side
        return crossings, next_track_id

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
