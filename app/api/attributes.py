import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import func, select

from app.api.deps import AppSettings, DBSession
from app.models.media import PersonCrop
from app.models.vectors import VectorIndexJob
from app.schemas.attributes import (
    ObjectType,
    PersonCropAttributeResponse,
    StructuredAnalyzeResponse,
)
from app.services.appearance_attributes import AppearanceAttributeService
from app.services.attribute_backfill import DurableAttributeBackfillService
from app.services.observation_index import ObservationIndexService
from app.services.stature import StatureService
from app.services.structured_attributes import StructuredAttributeService
from app.services.vlm import VLMRuntimeError

router = APIRouter(prefix="/attributes", tags=["attributes"])
UploadImage = Annotated[UploadFile, File(...)]


@router.post("/jobs/{crop_id}/retry")
def retry_attribute_job(crop_id: uuid.UUID, db: DBSession, settings: AppSettings) -> dict[str, str]:
    from app.services.vector_index_queue import (
        ATTRIBUTE_TARGET,
        VectorQueueFullError,
        attribute_queue,
    )

    if not attribute_queue.target_enabled(ATTRIBUTE_TARGET, settings):
        raise HTTPException(status_code=409, detail="Background VLM worker is disabled")
    crop = db.get(PersonCrop, crop_id)
    if crop is None:
        raise HTTPException(status_code=404, detail="Person crop not found")
    if (crop.attributes or {}).get("source") == "vlm":
        return {"status": "already_completed"}
    try:
        attribute_queue.enqueue_in_session(db, ATTRIBUTE_TARGET, crop_id, settings)
        db.commit()
    except VectorQueueFullError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Attribute queue is full; retry later") from exc
    attribute_queue.wake(settings)
    return {"status": "queued"}


@router.get("/jobs")
def attribute_jobs(db: DBSession) -> dict[str, object]:
    """Live coverage plus durable failures; an old backfill checkpoint is not live coverage."""
    pending = db.scalar(
        select(func.count())
        .select_from(PersonCrop)
        .where(
            func.coalesce(PersonCrop.attributes["source"].as_string(), "") != "vlm",
        )
    )
    counts = db.execute(
        select(VectorIndexJob.status, func.count())
        .where(
            VectorIndexJob.target == "person_attributes",
        )
        .group_by(VectorIndexJob.status)
    )
    failures = db.scalars(
        select(VectorIndexJob)
        .where(
            VectorIndexJob.target == "person_attributes",
            VectorIndexJob.status == "failed",
        )
        .order_by(VectorIndexJob.updated_at.desc())
        .limit(50)
    )
    return {
        "pending_crops": int(pending or 0),
        "queue": dict(counts.all()),
        "failures": [
            {"crop_id": str(job.object_id), "attempts": job.attempts, "error": job.last_error}
            for job in failures
        ],
    }


@router.post("/analyze", response_model=StructuredAnalyzeResponse)
def analyze_attributes(
    db: DBSession,
    settings: AppSettings,
    file: UploadImage,
    object_type: Annotated[ObjectType, Form()] = "person",
    bbox_json: Annotated[str | None, Form()] = None,
) -> StructuredAnalyzeResponse:
    service = StructuredAttributeService(db, settings)
    try:
        item = service.analyze_file(
            file.file,
            filename=file.filename or "upload.jpg",
            object_type=object_type,
            bbox=service.parse_bbox_json(bbox_json),
        )
    except (ValueError, VLMRuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StructuredAnalyzeResponse(count=1, items=[item])


@router.post(
    "/person-crops/{crop_id}/analyze",
    response_model=PersonCropAttributeResponse,
)
def analyze_person_crop_attributes(
    crop_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
    persist: bool = Query(default=True),
) -> PersonCropAttributeResponse:
    crop = db.get(PersonCrop, crop_id)
    if crop is None:
        raise HTTPException(status_code=404, detail="Person crop not found")
    try:
        attributes = StructuredAttributeService(db, settings).analyze_person_crop(
            crop,
            persist=persist,
        )
    except VLMRuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PersonCropAttributeResponse(crop_id=str(crop_id), attributes=attributes)


@router.post("/person-crops/tone-backfill")
def backfill_clothing_tone(
    db: DBSession,
    settings: AppSettings,
    limit: int = Query(default=500, ge=1, le=5000),
    force: bool = Query(default=False),
) -> dict[str, object]:
    """Read clothing tone with the durable keyset walker used by VLM backfill.

    The endpoint keeps its historical one-batch response and ``force`` semantics, but no longer
    loads an unbounded, newest-first query.  Existing VLM descriptions are visited and counted as
    skipped unless ``force=true``; this preserves the old response fields while the checkpoint
    lets repeated calls make progress through a large history.
    """

    service = AppearanceAttributeService(
        saturation_floor=settings.appearance_tone_saturation_floor,
        hue_value_floor=settings.appearance_tone_hue_value_floor,
        dark_ratio=settings.appearance_tone_dark_ratio,
    )
    stature_service = StatureService(db, settings)
    observations = ObservationIndexService(db, settings)

    def process(crop: PersonCrop) -> str:
        existing = crop.attributes or {}
        if (
            isinstance(existing, dict)
            and existing
            and existing.get("source") != "cv_tone"
            and not force
        ):
            return "skipped"
        path = _crop_path(settings, crop)
        attributes = service.describe(path) if path else None
        if attributes is None:
            return "unreadable"
        stature = stature_service.describe(crop.bbox, crop.camera_id)
        if stature:
            attributes["stature"] = stature
        crop.attributes = attributes
        db.add(crop)
        db.flush()
        observations.upsert_crop(crop)
        return "updated"

    durable = DurableAttributeBackfillService(
        db,
        settings,
        state_path=settings.data_dir
        / "tasks"
        / ("tone-backfill-force.json" if force else "tone-backfill.json"),
    )
    before = durable.load_progress()
    progress = durable.run(
        batch_size=limit,
        force=force,
        processor=process,
        include_described=True,
        max_items=limit,
    )
    return {
        "requested": limit,
        "seen": progress.attempted - before.attempted,
        "updated": progress.updated - before.updated,
        "skipped_described": progress.skipped - before.skipped,
        "unreadable": progress.unreadable - before.unreadable,
    }


def _crop_path(settings, crop: PersonCrop) -> Path | None:
    prefix = "/data/"
    if not crop.crop_url or not crop.crop_url.startswith(prefix):
        return None
    path = settings.data_dir / Path(crop.crop_url.removeprefix(prefix))
    return path if path.exists() else None


@router.post("/person-crops/backfill")
def backfill_person_crop_attributes(
    db: DBSession,
    settings: AppSettings,
    limit: int = Query(default=50, ge=1, le=5000),
    force: bool = Query(default=False),
) -> dict[str, object]:
    """Compatibility HTTP entry point backed by the durable VLM walker.

    Keep the historical response shape for callers, but use the same bounded keyset/checkpoint
    implementation as the worker.  Repeating the request resumes from the normal
    ``data/tasks/attribute-backfill.json`` checkpoint (or the separate
    ``attribute-backfill-force.json`` checkpoint for ``force=true``) instead of scanning the
    whole crop table again.
    """

    durable = DurableAttributeBackfillService(
        db,
        settings,
        state_path=settings.data_dir
        / "tasks"
        / ("attribute-backfill-force.json" if force else "attribute-backfill.json"),
    )
    before = durable.load_progress()
    progress = durable.run(
        batch_size=limit,
        force=force,
        max_items=limit,
    )
    return {
        "requested": limit,
        "force": force,
        "seen": progress.attempted - before.attempted,
        "updated": progress.updated - before.updated,
        "errors": progress.last_errors,
    }
