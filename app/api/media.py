import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import AppSettings, DBSession
from app.models.media import PersonCrop
from app.schemas.media import (
    ImageRead,
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


@router.post("/images/upload", response_model=ImageRead)
def upload_image(
    db: DBSession,
    settings: AppSettings,
    file: UploadImage,
    source_type: str = "upload",
    camera_id: uuid.UUID | None = None,
    location_id: uuid.UUID | None = None,
    captured_at: datetime | None = None,
) -> ImageRead:
    storage = StorageService(settings)
    image_url = storage.save_upload(file)
    try:
        return MediaService(db, settings).create_image_from_url(
            image_url=image_url,
            source_type=source_type,
            camera_id=camera_id,
            location_id=location_id,
            captured_at=captured_at,
        )
    except VectorQueueFullError as exc:
        db.rollback()
        storage.remove_data_url(image_url)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        db.rollback()
        storage.remove_data_url(image_url)
        raise


@router.get("/images", response_model=list[ImageRead])
def list_images(
    db: DBSession,
    settings: AppSettings,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    has_crops: bool = False,
) -> list[ImageRead]:
    return MediaService(db, settings).list_images(
        limit=limit,
        has_crops=has_crops,
        offset=offset,
    )


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
) -> list[PersonCropRead]:
    image = MediaService(db, settings).get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        return FrameProcessingService(db, settings).process_image(image)
    except VectorQueueFullError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/videos/upload", response_model=VideoProcessResponse)
def upload_video(
    db: DBSession,
    settings: AppSettings,
    file: UploadImage,
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
    try:
        frames = service.stream_mjpeg_frames(
            stream.stream_url,
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
