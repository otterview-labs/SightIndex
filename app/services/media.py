import logging
import time
import uuid
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.orm import Session, selectinload

from app.config.settings import Settings
from app.models.events import CountingEvent, RecognitionEvent
from app.models.media import Image, ImageTag, PersonCrop, PersonObservationIndex, VideoStream
from app.models.reid import ReidMatchFeedback
from app.models.vectors import CropFaceExtraction, FaceEmbedding, VectorIndexJob, VLEmbedding
from app.schemas.media import VideoStreamCreate
from app.services.opencv_capture import open_video_capture
from app.services.storage import VIDEO_SUFFIXES, StorageService
from app.services.time_utils import database_datetime, local_now

logger = logging.getLogger(__name__)

MAX_IMAGE_TAGS = 20
MAX_IMAGE_TAG_LENGTH = 64


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
        folderid: str | None = None,
        typenum: str | None = None,
        original_file_name: str | None = None,
        image_size: int | None = None,
        tags: list[str] | None = None,
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
            folderid=folderid,
            typenum=typenum,
            original_file_name=original_file_name,
            image_size=image_size,
        )
        self.db.add(image)
        self.db.flush()
        self._add_image_tags(image.id, tags)
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
        tag: str | None = None,
    ) -> list[Image]:
        stmt = select(Image).options(selectinload(Image.image_tags))
        if has_crops:
            stmt = stmt.join(PersonCrop, PersonCrop.image_id == Image.id).distinct()
        if tag:
            stmt = stmt.join(ImageTag).where(ImageTag.name == self.normalize_image_tag(tag))
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

    def image_count(self, tag: str | None = None) -> int:
        stmt = select(func.count(Image.id))
        if tag:
            stmt = stmt.join(ImageTag).where(ImageTag.name == self.normalize_image_tag(tag))
        return int(self.db.scalar(stmt) or 0)

    def get_image(self, image_id: uuid.UUID) -> Image | None:
        return self.db.scalar(
            select(Image).options(selectinload(Image.image_tags)).where(Image.id == image_id)
        )

    def replace_image_tags(self, image: Image, tags: list[str]) -> Image:
        self.db.execute(delete(ImageTag).where(ImageTag.image_id == image.id))
        self._add_image_tags(image.id, tags)
        self.db.commit()
        self.db.refresh(image)
        self.db.expire(image, ["image_tags"])
        return image

    @staticmethod
    def normalize_image_tag(value: str) -> str:
        normalized = " ".join(value.strip().split()).casefold()
        if not normalized:
            raise ValueError("Image tags cannot be empty")
        if len(normalized) > MAX_IMAGE_TAG_LENGTH:
            raise ValueError(f"Image tags cannot exceed {MAX_IMAGE_TAG_LENGTH} characters")
        return normalized

    @classmethod
    def normalize_image_tags(cls, tags: list[str] | None) -> list[str]:
        if not tags:
            return []
        if len(tags) > MAX_IMAGE_TAGS:
            raise ValueError(f"An image can have at most {MAX_IMAGE_TAGS} tags")
        normalized: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            value = cls.normalize_image_tag(tag)
            if value not in seen:
                normalized.append(value)
                seen.add(value)
        return normalized

    def _add_image_tags(self, image_id: uuid.UUID, tags: list[str] | None) -> None:
        self.db.add_all(
            ImageTag(image_id=image_id, name=tag) for tag in self.normalize_image_tags(tags)
        )

    def delete_image(self, image: Image) -> None:
        """Delete an image and every record or file derived from it."""

        self._delete_images([image])

    def delete_video(self, video_url: str) -> int:
        """Delete an uploaded source video and all frames derived from it."""

        if not self._is_uploaded_video_url(video_url):
            raise ValueError("video_url must be a supported file under /data/videos/")
        images = list(self.db.scalars(select(Image).where(Image.video_url == video_url)))
        self._delete_images(images)
        self.storage.remove_data_url(video_url)
        return len(images)

    def _delete_images(self, images: list[Image]) -> None:
        if not images:
            return

        image_ids = [image.id for image in images]
        file_urls = [
            url for image in images for url in (image.image_url, image.thumbnail_url)
        ]
        crops = list(
            self.db.scalars(select(PersonCrop).where(PersonCrop.image_id.in_(image_ids)))
        )
        file_urls.extend(crop.crop_url for crop in crops)
        crop_ids = [crop.id for crop in crops]
        object_ids = [*image_ids, *crop_ids]
        recognition_ids = list(
            self.db.scalars(
                select(RecognitionEvent.id).where(
                    or_(
                        RecognitionEvent.image_id.in_(image_ids),
                        RecognitionEvent.crop_id.in_(crop_ids) if crop_ids else False,
                    )
                )
            )
        )

        # Some deployed databases retain NO ACTION constraints, so preserve the cleanup
        # ordering used by scripts/cleanup_media_before.py.
        self.db.execute(
            update(VideoStream)
            .where(VideoStream.last_frame_image_id.in_(image_ids))
            .values(last_frame_image_id=None)
        )
        if crop_ids:
            self.db.execute(
                delete(ReidMatchFeedback).where(
                    or_(
                        ReidMatchFeedback.query_crop_id.in_(crop_ids),
                        ReidMatchFeedback.candidate_crop_id.in_(crop_ids),
                    )
                )
            )
        self.db.execute(
            delete(PersonObservationIndex).where(
                or_(
                    PersonObservationIndex.image_id.in_(image_ids),
                    PersonObservationIndex.crop_id.in_(crop_ids) if crop_ids else False,
                )
            )
        )
        self.db.execute(delete(VectorIndexJob).where(VectorIndexJob.object_id.in_(object_ids)))
        if crop_ids:
            self.db.execute(
                delete(CropFaceExtraction).where(CropFaceExtraction.crop_id.in_(crop_ids))
            )
        self.db.execute(
            delete(CountingEvent).where(
                or_(
                    CountingEvent.image_id.in_(image_ids),
                    CountingEvent.crop_id.in_(crop_ids) if crop_ids else False,
                    CountingEvent.recognition_event_id.in_(recognition_ids)
                    if recognition_ids
                    else False,
                )
            )
        )
        self.db.execute(
            delete(FaceEmbedding).where(
                or_(
                    FaceEmbedding.image_id.in_(image_ids),
                    FaceEmbedding.crop_id.in_(crop_ids) if crop_ids else False,
                )
            )
        )
        self.db.execute(delete(VLEmbedding).where(VLEmbedding.object_id.in_(object_ids)))
        self.db.execute(delete(ImageTag).where(ImageTag.image_id.in_(image_ids)))
        if recognition_ids:
            self.db.execute(
                delete(RecognitionEvent).where(RecognitionEvent.id.in_(recognition_ids))
            )
        if crop_ids:
            self.db.execute(delete(PersonCrop).where(PersonCrop.id.in_(crop_ids)))
        self.db.execute(delete(Image).where(Image.id.in_(image_ids)))
        self.db.commit()

        for url in file_urls:
            if url:
                self.storage.remove_data_url(url)
        self._remove_deleted_vectors("image", image_ids)
        self._remove_deleted_vectors("person_crop", crop_ids)
        self._remove_deleted_vectors("reid_person_crop", crop_ids)

    @staticmethod
    def _is_uploaded_video_url(video_url: str) -> bool:
        return (
            video_url.startswith("/data/videos/")
            and Path(video_url).suffix.lower() in VIDEO_SUFFIXES
        )

    def _remove_deleted_vectors(self, object_type: str, object_ids: list[uuid.UUID]) -> None:
        if not object_ids or not self.settings.milvus_enabled:
            return
        try:
            from app.services.vector_index import MilvusVectorIndex

            collection = MilvusVectorIndex(self.settings)._collection(object_type)
            quoted_ids = ", ".join(f'"{object_id}"' for object_id in object_ids)
            collection.delete(expr=f"object_id in [{quoted_ids}]")
            collection.flush()
        except Exception:
            # SQL is already committed. A later index rebuild can clear stale external vectors.
            logger.warning(
                "Could not remove deleted %s vectors from Milvus", object_type, exc_info=True
            )

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
