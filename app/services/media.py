import time
import uuid
from collections.abc import Iterator
from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.models.events import CountingEvent
from app.models.media import Image, PersonCrop, VideoStream
from app.schemas.media import VideoStreamCreate
from app.services.opencv_capture import open_video_capture
from app.services.storage import StorageService
from app.services.time_utils import database_datetime, local_now


class MediaService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.storage = StorageService(settings)

    def create_image_from_url(
        self,
        image_url: str,
        source_type: str = "upload",
        camera_id: uuid.UUID | None = None,
        location_id: uuid.UUID | None = None,
        captured_at: datetime | None = None,
    ) -> Image:
        image = Image(
            image_url=image_url,
            source_type=source_type,
            camera_id=camera_id,
            location_id=location_id,
            captured_at=(
                database_datetime(
                    captured_at,
                    self.settings,
                    self.db.get_bind().dialect.name,
                )
                if captured_at is not None
                else None
            ),
        )
        self.db.add(image)
        self.db.flush()
        enqueued = self._enqueue_image_job_in_session(image)
        self.db.commit()
        self.db.refresh(image)
        if enqueued:
            from app.services.vector_index_queue import vector_index_queue

            vector_index_queue.wake(self.settings)
        elif self.settings.vector_index_on_ingest:
            self._try_index_image(image)
        return image

    def list_images(
        self,
        limit: int = 50,
        has_crops: bool = False,
        offset: int = 0,
    ) -> list[Image]:
        stmt = select(Image)
        if has_crops:
            stmt = stmt.join(PersonCrop, PersonCrop.image_id == Image.id).distinct()
        stmt = stmt.order_by(Image.created_at.desc()).offset(offset).limit(limit)
        return list(self.db.scalars(stmt))

    def media_counts(self) -> dict[str, int]:
        image_with_crops = self.db.scalar(
            select(func.count(func.distinct(PersonCrop.image_id)))
        )
        person_crops = self.db.scalar(select(func.count()).select_from(PersonCrop))
        return {
            "image_with_crops_count": int(image_with_crops or 0),
            "person_crop_count": int(person_crops or 0),
        }

    def get_image(self, image_id: uuid.UUID) -> Image | None:
        return self.db.get(Image, image_id)

    def list_person_crops(
        self,
        image_id: uuid.UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PersonCrop]:
        stmt = (
            select(PersonCrop)
            .order_by(PersonCrop.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        if image_id:
            stmt = stmt.where(PersonCrop.image_id == image_id)
        return list(self.db.scalars(stmt))

    def create_stream(self, payload: VideoStreamCreate) -> VideoStream:
        data = payload.model_dump()
        line = data.get("counting_line")
        if line and (line["x1"], line["y1"]) == (line["x2"], line["y2"]):
            raise ValueError("Counting line must have two different points")
        stream = VideoStream(**data, status="stopped")
        self.db.add(stream)
        self.db.commit()
        self.db.refresh(stream)
        return stream

    def _try_index_image(self, image: Image) -> None:
        if not self.settings.vector_index_on_ingest:
            return
        if self.settings.vector_index_on_ingest_background:
            return  # background jobs are staged atomically with the Image transaction
        try:
            from app.services.vector_index import VectorIndexingService

            indexer = VectorIndexingService(self.db, self.settings)
            # Keep the same durable ordering as the worker: external write + flush first, then
            # commit the SQL marker. FastAPI only closes the dependency Session, so this commit
            # is required in the legacy synchronous mode.
            indexer.write_image_vector(image, flush=False)
            indexer.index.flush("image")
            indexer.record_image_index(image)
            self.db.commit()
        except Exception:
            self.db.rollback()
            return

    def _enqueue_image_job_in_session(self, image: Image) -> bool:
        if not (
            self.settings.vector_index_on_ingest
            and self.settings.vector_index_on_ingest_background
        ):
            return False
        from app.services.vector_index_queue import vector_index_queue

        return vector_index_queue.enqueue_in_session(
            self.db,
            "image",
            image.id,
            self.settings,
        )

    def list_streams(self, limit: int = 50) -> list[VideoStream]:
        stmt = select(VideoStream).order_by(VideoStream.created_at.desc()).limit(limit)
        return list(self.db.scalars(stmt))

    def get_stream(self, stream_id: uuid.UUID) -> VideoStream | None:
        return self.db.get(VideoStream, stream_id)

    def delete_stream(self, stream: VideoStream) -> None:
        self.db.execute(
            update(CountingEvent)
            .where(CountingEvent.stream_id == stream.id)
            .values(stream_id=None)
        )
        self.db.delete(stream)
        self.db.commit()

    def capture_stream_snapshot(self, stream: VideoStream) -> Image:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception as exc:
            raise ValueError(f"OpenCV is not installed: {exc}") from exc

        capture = open_video_capture(cv2, stream.stream_url, self.settings)
        if not capture.isOpened():
            raise ValueError("Could not open stream")
        try:
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Failed to read frame")
            self.settings.frames_dir.mkdir(parents=True, exist_ok=True)
            now = local_now(self.settings)
            filename = f"{stream.id}_{now.strftime('%Y%m%d%H%M%S%f')}_snapshot.jpg"
            path = self.settings.frames_dir / filename
            if not cv2.imwrite(
                str(path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.settings.frame_jpeg_quality)],
            ):
                raise ValueError("Failed to write snapshot")
            image = Image(
                image_url=f"/data/frames/{filename}",
                source_type="stream_frame",
                camera_id=stream.camera_id,
                location_id=stream.location_id,
                captured_at=now,
            )
            self.db.add(image)
            self.db.flush()
            stream.last_frame_image_id = image.id
            self.db.add(stream)
            try:
                enqueued = self._enqueue_image_job_in_session(image)
                self.db.commit()
            except Exception:
                self.db.rollback()
                path.unlink(missing_ok=True)
                raise
            self.db.refresh(image)
            if enqueued:
                from app.services.vector_index_queue import vector_index_queue

                vector_index_queue.wake(self.settings)
            elif self.settings.vector_index_on_ingest:
                self._try_index_image(image)
            return image
        finally:
            capture.release()

    def stream_mjpeg_frames(
        self,
        stream_url: str,
        fps: float = 6.0,
        jpeg_quality: int = 92,
    ) -> Iterator[bytes]:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception as exc:
            raise ValueError(f"OpenCV is not installed: {exc}") from exc

        capture = open_video_capture(cv2, stream_url, self.settings)
        if not capture.isOpened():
            raise ValueError("Could not open stream")

        interval = 1 / max(fps, 0.2)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

        def generate() -> Iterator[bytes]:
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    encoded, buffer = cv2.imencode(".jpg", frame, encode_params)
                    if not encoded:
                        continue
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-cache\r\n\r\n"
                        + buffer.tobytes()
                        + b"\r\n"
                    )
                    time.sleep(interval)
            finally:
                capture.release()

        return generate()

    def update_stream_status(
        self,
        stream: VideoStream,
        status: str,
        last_error: str | None = None,
        last_frame_image_id: uuid.UUID | None = None,
    ) -> VideoStream:
        now = local_now(self.settings)
        stream.status = status
        stream.last_error = last_error
        if status == "running":
            stream.started_at = stream.started_at or now
            stream.stopped_at = None
        if status == "stopped":
            stream.stopped_at = now
        if last_frame_image_id:
            stream.last_frame_image_id = last_frame_image_id
        self.db.add(stream)
        self.db.commit()
        self.db.refresh(stream)
        return stream

    def update_stream_counting_line(
        self,
        stream: VideoStream,
        counting_line: dict[str, float] | None,
    ) -> VideoStream:
        if counting_line and (
            counting_line["x1"],
            counting_line["y1"],
        ) == (
            counting_line["x2"],
            counting_line["y2"],
        ):
            raise ValueError("Counting line must have two different points")
        stream.counting_line = counting_line
        self.db.add(stream)
        self.db.commit()
        self.db.refresh(stream)
        return stream
