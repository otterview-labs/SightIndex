import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import select

from app.api.deps import AppSettings, DBSession
from app.models.media import PersonCrop
from app.schemas.attributes import (
    ObjectType,
    PersonCropAttributeResponse,
    StructuredAnalyzeResponse,
)
from app.services.appearance_attributes import AppearanceAttributeService
from app.services.observation_index import ObservationIndexService
from app.services.stature import StatureService
from app.services.structured_attributes import StructuredAttributeService
from app.services.vlm import VLMRuntimeError

router = APIRouter(prefix="/attributes", tags=["attributes"])
UploadImage = Annotated[UploadFile, File(...)]


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
    """Reads clothing tone off crops that have no attributes yet.

    Skips anything a VLM already described, unless forced: this reader knows brightness and
    little else, and replacing a real description with it would be a downgrade.
    """

    service = AppearanceAttributeService(
        saturation_floor=settings.appearance_tone_saturation_floor,
        hue_value_floor=settings.appearance_tone_hue_value_floor,
        dark_ratio=settings.appearance_tone_dark_ratio,
    )
    stature_service = StatureService(db, settings)
    observations = ObservationIndexService(db, settings)
    crops = list(
        db.scalars(select(PersonCrop).order_by(PersonCrop.captured_at.desc()).limit(limit))
    )
    seen = updated = skipped = unreadable = 0
    for crop in crops:
        seen += 1
        existing = crop.attributes or {}
        if existing and existing.get("source") != "cv_tone" and not force:
            skipped += 1
            continue
        path = _crop_path(settings, crop)
        attributes = service.describe(path) if path else None
        if attributes is None:
            unreadable += 1
            continue
        stature = stature_service.describe(crop.bbox, crop.camera_id)
        if stature:
            attributes["stature"] = stature
        crop.attributes = attributes
        db.add(crop)
        db.flush()
        observations.upsert_crop(crop)
        updated += 1
    db.commit()
    return {
        "requested": limit,
        "seen": seen,
        "updated": updated,
        "skipped_described": skipped,
        "unreadable": unreadable,
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
    seen, updated, errors = StructuredAttributeService(
        db,
        settings,
    ).analyze_unparsed_person_crops(limit, force=force)
    return {"requested": limit, "force": force, "seen": seen, "updated": updated, "errors": errors}
