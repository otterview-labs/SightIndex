import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.api.deps import AppSettings, DBSession
from app.models.media import PersonCrop
from app.schemas.events import RecognitionEventRead
from app.schemas.media import PersonCropRead
from app.schemas.persons import (
    FaceEmbeddingRead,
    PersonCreate,
    PersonRead,
    PersonTrajectoryResponse,
)
from app.services.faces import FaceRecognitionService
from app.services.persons import PersonService

router = APIRouter(prefix="/persons", tags=["persons"])
UploadImage = Annotated[UploadFile, File(...)]


@router.post("", response_model=PersonRead)
def create_person(payload: PersonCreate, db: DBSession) -> PersonRead:
    return PersonService(db).create(payload)


@router.get("", response_model=list[PersonRead])
def list_persons(
    db: DBSession,
    query: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[PersonRead]:
    return PersonService(db).list(query=query, limit=limit)


@router.get("/{person_id}", response_model=PersonRead)
def get_person(person_id: uuid.UUID, db: DBSession) -> PersonRead:
    person = PersonService(db).get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


@router.post("/{person_id}/faces", response_model=FaceEmbeddingRead)
def add_person_face(
    person_id: uuid.UUID,
    file: UploadImage,
    db: DBSession,
    settings: AppSettings,
) -> FaceEmbeddingRead:
    person = PersonService(db).get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    try:
        return FaceRecognitionService(db, settings).enroll_person_face(person, file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{person_id}/faces/from-crop/{crop_id}", response_model=FaceEmbeddingRead)
def add_person_face_from_crop(
    person_id: uuid.UUID,
    crop_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
) -> FaceEmbeddingRead:
    person = PersonService(db).get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    crop = db.get(PersonCrop, crop_id)
    if crop is None:
        raise HTTPException(status_code=404, detail="Person crop not found")
    try:
        return FaceRecognitionService(db, settings).enroll_person_face_from_crop(person, crop)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{person_id}/crops/{crop_id}", response_model=PersonCropRead)
def label_person_crop(
    person_id: uuid.UUID,
    crop_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
) -> PersonCropRead:
    """Marks a body crop as this person, for ReID to use as a gallery seed.

    The face route does this too, but only for crops with a findable face. This one asks nothing
    of the image, so a person can be named from behind, at distance, or with face recognition
    switched off entirely.
    """

    person = PersonService(db, settings).get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    crop = db.get(PersonCrop, crop_id)
    if crop is None:
        raise HTTPException(status_code=404, detail="Person crop not found")
    if crop.person_id is not None and crop.person_id != person_id:
        raise HTTPException(
            status_code=409,
            detail="This crop is already labelled as another person; remove that label first",
        )
    return PersonCropRead.model_validate(PersonService(db, settings).label_crop(person, crop))


@router.delete("/{person_id}/crops/{crop_id}", response_model=PersonCropRead)
def unlabel_person_crop(
    person_id: uuid.UUID,
    crop_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
) -> PersonCropRead:
    crop = db.get(PersonCrop, crop_id)
    if crop is None:
        raise HTTPException(status_code=404, detail="Person crop not found")
    if crop.person_id != person_id:
        raise HTTPException(status_code=404, detail="This crop is not labelled as that person")
    return PersonCropRead.model_validate(PersonService(db, settings).unlabel_crop(crop))


@router.get("/{person_id}/crops", response_model=list[PersonCropRead])
def list_person_crops(
    person_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[PersonCropRead]:
    service = PersonService(db, settings)
    if service.get(person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return [
        PersonCropRead.model_validate(crop) for crop in service.labelled_crops(person_id, limit)
    ]


@router.get("/{person_id}/faces", response_model=list[FaceEmbeddingRead])
def list_person_faces(
    person_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[FaceEmbeddingRead]:
    person = PersonService(db).get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return FaceRecognitionService(db, settings).list_person_faces(person_id, limit=limit)


@router.get("/{person_id}/events", response_model=list[RecognitionEventRead])
def get_person_events(
    person_id: uuid.UUID,
    db: DBSession,
    limit: int = Query(default=100, ge=1, le=1000),
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    min_similarity: float | None = Query(default=None, ge=0.0, le=1.0),
) -> list[RecognitionEventRead]:
    service = PersonService(db)
    person = service.get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return service.events(
        person_id=person_id,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
        min_similarity=min_similarity,
    )


@router.get("/{person_id}/trajectory", response_model=PersonTrajectoryResponse)
def get_person_trajectory(
    person_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
    limit: int = Query(default=100, ge=1, le=1000),
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    min_similarity: float | None = Query(default=None, ge=0.0, le=1.0),
    mode: Literal["all", "face", "vector", "reid"] = Query(default="all"),
    backfill_missing: bool = False,
) -> PersonTrajectoryResponse:
    service = PersonService(db, settings)
    person = service.get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    try:
        items, warnings = service.trajectory_report(
            person=person,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            min_similarity=min_similarity,
            mode=mode,
            backfill_missing=backfill_missing,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PersonTrajectoryResponse(person=person, items=items, warnings=warnings)
