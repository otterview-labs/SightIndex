import uuid
from datetime import datetime

from pydantic import BaseModel

from app.schemas.common import ORMModel


class RecognitionEventRead(ORMModel):
    id: uuid.UUID
    image_id: uuid.UUID | None
    crop_id: uuid.UUID | None
    person_id: uuid.UUID | None
    unknown_cluster_id: uuid.UUID | None
    camera_id: uuid.UUID | None
    location_id: uuid.UUID | None
    confidence: float | None
    similarity: float | None
    face_bbox: dict[str, float] | None = None
    result_type: str
    recognized_at: datetime
    created_at: datetime


class PersonTrajectoryPoint(BaseModel):
    event_id: uuid.UUID | None = None
    counting_event_id: uuid.UUID | None = None
    image_id: uuid.UUID | None
    crop_id: uuid.UUID | None
    person_id: uuid.UUID
    person_name: str
    camera_id: uuid.UUID | None
    camera_name: str | None = None
    location_id: uuid.UUID | None
    location_name: str | None = None
    similarity: float | None
    vector_score: float | None = None
    confidence: float | None
    face_bbox: dict[str, float] | None = None
    match_source: str = "face"
    result_type: str
    recognized_at: datetime
    image_url: str | None = None
    crop_url: str | None = None


class CountSummary(BaseModel):
    recognition_event_count: int
    counting_event_count: int
    unique_person_count: int
    unique_unknown_count: int
    image_count: int = 0
    person_crop_count: int = 0
