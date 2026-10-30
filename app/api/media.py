import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import func, select

from app.api.deps import AppSettings, DBSession
from app.models.media import Image, PersonCrop
from app.schemas.media import (
    ImageDeletionCounts,
    ImageDeletionResponse,
    ImageRead,
    ImageTagSearchResponse,
    ImageTagsUpdate,
    ImageVideoPosition,
    MediaCounts,
    PersonCropRead,
    StreamActionResponse,
    VideoProcessResponse,
    VideoStreamCountingLineUpdate,
    VideoStreamCreate,
    VideoStreamRead,
)
from app.services.frame_processing import FrameProcessingService
from app.services.media import MediaService
from app.services.media_deletion import MediaDeletionService, MediaDeletionUnavailable
from app.services.statistics import StatisticsService
from app.services.storage import StorageService
from app.services.stream_runtime import stream_runtime
from app.services.vector_index_queue import VectorQueueFullError
from app.services.video_processing import (
    VideoProcessingBackpressureError,
    VideoProcessingService,
)

router = APIRouter(tags=["media"])
UploadImage = Annotated[UploadFile, File(...)]
logger = logging.getLogger(__name__)


def uploaded_file_size(file: UploadFile) -> int | None:
    """Return the original multipart file size without relying on a client header."""
    if file.size is not None:
        return file.size
    try:
        position = file.file.tell()
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(position)
        return size
    except (AttributeError, OSError):
        return None


@router.post("/images/upload", response_model=ImageRead)
def upload_image(
    db: DBSession,
    settings: AppSettings,
    file: UploadImage,
    source_type: Annotated[str, Form()] = "upload",
    camera_id: Annotated[uuid.UUID | None, Form()] = None,
    location_id: Annotated[uuid.UUID | None, Form()] = None,
    captured_at: Annotated[datetime | None, Form()] = None,
    folderid: Annotated[str | None, Form()] = None,
    typenum: Annotated[str | None, Form()] = None,
    tags: Annotated[list[str] | None, Form()] = None,
) -> ImageRead:
    storage = StorageService(settings)
    original_file_name = file.filename or None
    image_size = uploaded_file_size(file)
    image_url = storage.save_upload(file)
    try:
        return MediaService(db, settings).create_image_from_url(
            image_url=image_url,
            source_type=source_type,
            camera_id=camera_id,
            location_id=location_id,
            captured_at=captured_at,
            folderid=folderid,
            typenum=typenum,
            original_file_name=original_file_name,
            image_size=image_size,
            tags=tags,
        )
    except VectorQueueFullError as exc:
        db.rollback()
        storage.remove_data_url(image_url)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        db.rollback()
        storage.remove_data_url(image_url)
        raise


@router.get("/images", response_model=list[ImageRead] | ImageTagSearchResponse)
def list_images(
    db: DBSession,
    settings: AppSettings,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    has_crops: bool = False,
    tag: str | None = Query(default=None, max_length=64),
) -> list[ImageRead] | ImageTagSearchResponse:
    try:
        service = MediaService(db, settings)
        items = service.list_images(
            limit=limit,
            has_crops=has_crops,
            offset=offset,
            tag=tag,
        )
        if tag:
            normalized_tag = service.normalize_image_tag(tag)
            return ImageTagSearchResponse(
                items=items,
                tag=normalized_tag,
                total=service.image_count(normalized_tag),
                limit=limit,
                offset=offset,
                notice=f'Exact tag match for "{normalized_tag}"',
            )
        return items
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/media/counts", response_model=MediaCounts)
def get_media_counts(db: DBSession, settings: AppSettings) -> MediaCounts:
    return MediaCounts.model_validate(MediaService(db, settings).media_counts())


@router.get("/images/{image_id}", response_model=ImageRead)
def get_image(
    image_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
) -> ImageRead:
    image = MediaService(db, settings).get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return image


@router.put("/images/{image_id}/tags", response_model=ImageRead)
def replace_image_tags(
    image_id: uuid.UUID,
    payload: ImageTagsUpdate,
    db: DBSession,
    settings: AppSettings,
) -> ImageRead:
    service = MediaService(db, settings)
    image = service.get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        return service.replace_image_tags(image, payload.tags)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/images/{image_id}", response_model=ImageDeletionResponse)
def delete_image(
    image_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
    dry_run: bool = Query(default=False),
    delete_files: bool = Query(default=True),
    reason: str | None = Query(default=None, max_length=200),
) -> ImageDeletionResponse:
    """Delete one image together with its crops, events, vectors and files.

    Idempotent: an image that is already gone returns 404 rather than a conflict. With
    ``dry_run`` nothing is written and ``deleted`` holds the counts that would be affected.
    Refuses with 503 when Milvus is configured but unreachable, because skipping the vector
    delete would leave search hits pointing at rows that no longer exist.
    """

    service = MediaDeletionService(db, settings)
    try:
        result = service.delete_image(
            image_id,
            dry_run=dry_run,
            delete_files=delete_files,
            reason=reason,
        )
    except MediaDeletionUnavailable as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return ImageDeletionResponse(
        image_id=result.image_id,
        dry_run=result.dry_run,
        deleted=ImageDeletionCounts(**result.counts),
        warnings=result.warnings,
    )


@router.delete("/videos", status_code=204)
def delete_video(
    video_url: str,
    db: DBSession,
    settings: AppSettings,
) -> Response:
    try:
        MediaService(db, settings).delete_video(video_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(status_code=204)


@router.get("/images/{image_id}/video-position", response_model=ImageVideoPosition)
def get_image_video_position(
    image_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
) -> ImageVideoPosition:
    """Resolve where a frame sits in its source video.

    Frames ingested after the video-url columns landed carry an exact link. Older rows
    fall back to the filename convention (<video-stem>_<timestamp>.jpg) plus the earliest
    stored frame of the same video, which is accurate to within one sampling interval.
    """

    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    if image.source_type != "video_frame":
        raise HTTPException(status_code=404, detail="Image is not a video frame")

    if image.video_url and image.video_offset_ms is not None:
        video_path = settings.data_dir / image.video_url.removeprefix("/data/")
        if video_path.is_file():
            return ImageVideoPosition(
                image_id=image.id,
                video_url=image.video_url,
                video_offset_ms=float(image.video_offset_ms),
                exact=True,
                source_image_url=image.image_url,
            )

    stem = Path(image.image_url).name.rsplit("_", 1)[0]
    candidates = sorted(
        path for path in settings.videos_dir.glob(f"{stem}.*") if path.is_file()
    )
    if not candidates:
        raise HTTPException(status_code=404, detail="Source video not found")
    earliest = db.scalar(
        select(func.min(Image.captured_at)).where(
            Image.source_type == "video_frame",
            # The "_" is a LIKE wildcard that also matches the literal separator in frame
            # filenames; stems are uuid hex so nothing else can collide.
            Image.image_url.like(f"/data/frames/{stem}%"),
        )
    )
    offset_ms = 0.0
    if image.captured_at is not None and earliest is not None:
        offset_ms = max(0.0, (image.captured_at - earliest).total_seconds() * 1000.0)
    return ImageVideoPosition(
        image_id=image.id,
        video_url=f"/data/videos/{candidates[0].name}",
        video_offset_ms=offset_ms,
        exact=False,
        source_image_url=image.image_url,
    )


@router.get("/person-crops", response_model=list[PersonCropRead])
def list_person_crops(
    db: DBSession,
    settings: AppSettings,
    image_id: uuid.UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[PersonCropRead]:
    return MediaService(db, settings).list_person_crops(
        image_id=image_id,
        limit=limit,
        offset=offset,
    )


@router.get("/person-crops/{crop_id}", response_model=PersonCropRead)
def get_person_crop(crop_id: uuid.UUID, db: DBSession) -> PersonCropRead:
    """One crop by id, so a page can show the image a search was run from."""

    crop = db.get(PersonCrop, crop_id)
    if crop is None:
        raise HTTPException(status_code=404, detail="Person crop not found")
    return PersonCropRead.model_validate(crop)


@router.post("/images/{image_id}/process", response_model=list[PersonCropRead])
def process_image(
    image_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
    force: bool = Query(default=False),
) -> list[PersonCropRead]:
    image = MediaService(db, settings).get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        return FrameProcessingService(db, settings).process_image(image, force=force)
    except VectorQueueFullError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/videos/upload", response_model=VideoProcessResponse)
def upload_video(
    db: DBSession,
    settings: AppSettings,
    file: UploadImage,
    folderid: Annotated[str | None, Form()] = None,
    original_file_name: Annotated[str | None, Form()] = None,
    image_size: Annotated[int | None, Form(ge=0)] = None,
    frame_interval_seconds: float = Query(default=1.0, ge=0.1, le=3600),
    max_frames: int = Query(default=120, ge=1, le=2000),
    store_empty_frames: bool | None = None,
    line_x1: float | None = Query(default=None, ge=0.0, le=1.0),
    line_y1: float | None = Query(default=None, ge=0.0, le=1.0),
    line_x2: float | None = Query(default=None, ge=0.0, le=1.0),
    line_y2: float | None = Query(default=None, ge=0.0, le=1.0),
    camera_id: uuid.UUID | None = None,
    location_id: uuid.UUID | None = None,
    captured_at: datetime | None = None,
) -> VideoProcessResponse:
    try:
        
        original_file_name = original_file_name or file.filename or None
        image_size = image_size if image_size is not None else uploaded_file_size(file)
        print(
            f"Video upload folderid: {folderid}  "
            f"original_file_name: {original_file_name}  image_size: {image_size}"
        )
        return VideoProcessingService(db, settings).process_upload(
            file=file,
            frame_interval_seconds=frame_interval_seconds,
            max_frames=max_frames,
            store_empty_frames=store_empty_frames,
            counting_line=VideoProcessingService.create_counting_line(
                line_x1,
                line_y1,
                line_x2,
                line_y2,
            ),
            camera_id=camera_id,
            location_id=location_id,
            captured_at=captured_at,
            folderid=folderid,
            original_file_name=original_file_name,
            image_size=image_size,
        )
    except VideoProcessingBackpressureError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=exc.api_detail()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/videos/{video_id}/clips")
def list_video_clips(video_id: uuid.UUID) -> dict[str, object]:
    return {"video_id": str(video_id), "items": []}


@router.post("/streams", response_model=VideoStreamRead)
def create_stream(
    payload: VideoStreamCreate,
    db: DBSession,
    settings: AppSettings,
) -> VideoStreamRead:
    try:
        return MediaService(db, settings).create_stream(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/streams", response_model=list[VideoStreamRead])
def list_streams(
    db: DBSession,
    settings: AppSettings,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[VideoStreamRead]:
    return MediaService(db, settings).list_streams(limit=limit)


@router.get("/streams/{stream_id}", response_model=VideoStreamRead)
def get_stream(
    stream_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
) -> VideoStreamRead:
    stream = MediaService(db, settings).get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found")
    return stream


@router.delete("/streams/{stream_id}", response_model=StreamActionResponse)
def delete_stream(
    stream_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
) -> StreamActionResponse:
    service = MediaService(db, settings)
    stream = service.get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found")
    message = stream_runtime.stop(stream_id)
    if not stream_runtime.wait_stopped(stream_id):
        raise HTTPException(status_code=409, detail="Stream is still stopping")
    service.delete_stream(stream)
    return StreamActionResponse(stream_id=stream_id, status="deleted", message=message)


@router.patch("/streams/{stream_id}/counting-line", response_model=VideoStreamRead)
def update_stream_counting_line(
    stream_id: uuid.UUID,
    payload: VideoStreamCountingLineUpdate,
    db: DBSession,
    settings: AppSettings,
) -> VideoStreamRead:
    service = MediaService(db, settings)
    stream = service.get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found")
    line = payload.counting_line.model_dump() if payload.counting_line else None
    try:
        return service.update_stream_counting_line(stream, line)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/streams/{stream_id}/counts")
def get_stream_counts(
    stream_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> dict[str, object]:
    stream = MediaService(db, settings).get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found")
    summary = StatisticsService(db).count_summary(
        start_time=start_time,
        end_time=end_time,
        stream_id=stream_id,
    )
    total_summary = StatisticsService(db).count_summary(
        start_time=start_time,
        end_time=end_time,
    )
    return {
        "stream_id": str(stream_id),
        "stream_name": stream.name,
        "counting_event_count": summary.counting_event_count,
        "total_counting_event_count": total_summary.counting_event_count,
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": end_time.isoformat() if end_time else None,
    }


@router.post("/streams/{stream_id}/snapshot", response_model=ImageRead)
def capture_stream_snapshot(
    stream_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
) -> ImageRead:
    service = MediaService(db, settings)
    stream = service.get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found")
    try:
        return service.capture_stream_snapshot(stream)
    except VectorQueueFullError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/streams/{stream_id}/mjpeg")
def stream_mjpeg_preview(
    stream_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
    fps: float = Query(default=6.0, ge=0.2, le=15.0),
    jpeg_quality: int = Query(default=92, ge=30, le=100),
) -> StreamingResponse:
    service = MediaService(db, settings)
    stream = service.get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found")
    stream_url = stream.stream_url
    db.close()
    try:
        frames = service.stream_mjpeg_frames(
            stream_url,
            fps=fps,
            jpeg_quality=jpeg_quality,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StreamingResponse(
        frames,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/streams/{stream_id}/start", response_model=StreamActionResponse)
def start_stream(
    stream_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
) -> StreamActionResponse:
    service = MediaService(db, settings)
    stream = service.get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found")
    if stream_runtime.is_running(stream_id):
        service.update_stream_status(stream, "running", last_error=None)
        return StreamActionResponse(
            stream_id=stream_id,
            status="running",
            message="already running",
        )
    service.update_stream_status(stream, "starting", last_error=None)
    message = stream_runtime.start(stream_id)
    return StreamActionResponse(stream_id=stream_id, status="starting", message=message)


@router.post("/streams/{stream_id}/stop", response_model=StreamActionResponse)
def stop_stream(
    stream_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
) -> StreamActionResponse:
    service = MediaService(db, settings)
    stream = service.get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found")
    message = stream_runtime.stop(stream_id)
    service.update_stream_status(stream, "stopped")
    return StreamActionResponse(stream_id=stream_id, status="stopped", message=message)
