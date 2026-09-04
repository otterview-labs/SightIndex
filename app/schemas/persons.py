import uuid
from datetime import datetime

from pydantic import BaseModel

from app.schemas.common import ORMModel
from app.schemas.events import PersonTrajectoryPoint


class PersonCreate(BaseModel):
    name: str
    employee_no: str | None = None
    phone: str | None = None
    department: str | None = None
    avatar_url: str | None = None
    status: str = "active"


class PersonRead(ORMModel):
    id: uuid.UUID
    name: str
    employee_no: str | None
    phone: str | None
    department: str | None
    avatar_url: str | None
    status: str
    created_at: datetime
    updated_at: datetime


class FaceEmbeddingRead(ORMModel):
    id: uuid.UUID
    person_id: uuid.UUID | None
    image_id: uuid.UUID | None
    crop_id: uuid.UUID | None
    face_bbox: dict[str, float] | None
    face_model: str
    quality_score: float | None
    created_at: datetime


class FaceMatchItem(BaseModel):
    person_id: uuid.UUID
    person_name: str
    face_embedding_id: uuid.UUID
    similarity: float
    quality_score: float | None = None
    image_id: uuid.UUID | None = None
    crop_id: uuid.UUID | None = None


class FaceRecognitionResponse(BaseModel):
    result_type: str
    person: PersonRead | None = None
    similarity: float | None = None
    threshold: float
    image_id: uuid.UUID | None = None
    crop_id: uuid.UUID | None = None
    event_id: uuid.UUID | None = None
    face_bbox: dict[str, float] | None = None
    matches: list[FaceMatchItem]


class FaceSearchResponse(BaseModel):
    image_id: uuid.UUID | None = None
    face_bbox: dict[str, float] | None = None
    matches: list[FaceMatchItem]


class FaceDiagnosticItem(BaseModel):
    crop_id: uuid.UUID
    image_id: uuid.UUID | None = None
    crop_url: str | None = None
    image_url: str | None = None
    captured_at: datetime | None = None
    existing_result_type: str | None = None
    existing_person_id: uuid.UUID | None = None
    existing_similarity: float | None = None
    detection_score: float | None = None
    face_bbox: dict[str, float] | None = None
    top_person_id: uuid.UUID | None = None
    top_person_name: str | None = None
    top_similarity: float | None = None
    threshold: float
    verdict: str
    reason: str
    can_enroll: bool = False


class FaceDiagnosticResponse(BaseModel):
    threshold: float
    items: list[FaceDiagnosticItem]


class FaceRecognitionRebuildResponse(BaseModel):
    requested: int
    seen: int
    skipped: int
    events_created: int
    events_updated: int = 0
    matched: int
    errors: list[str]


class FaceLibraryRebuildResponse(BaseModel):
    requested: int
    seen: int
    updated: int
    skipped: int
    errors: list[str]


class PersonTrajectoryResponse(BaseModel):
    person: PersonRead
    items: list[PersonTrajectoryPoint]
    # Why a mode returned nothing (ReID unreachable, no gallery seeds, ...). An empty list
    # still means "queried fine, no matches".
    warnings: list[str] = []
