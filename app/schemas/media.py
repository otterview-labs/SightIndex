import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel, SearchFilters


class ImageRead(ORMModel):
    id: uuid.UUID
    image_url: str
    thumbnail_url: str | None
    source_type: str
    camera_id: uuid.UUID | None
    location_id: uuid.UUID | None
    location_name: str | None = None
    captured_at: datetime | None
    created_at: datetime


class PersonCropRead(ORMModel):
    id: uuid.UUID
    image_id: uuid.UUID
    crop_url: str
    bbox: dict[str, Any]
    attributes: dict[str, Any] | None = None
    person_id: uuid.UUID | None
    camera_id: uuid.UUID | None
    location_id: uuid.UUID | None
    captured_at: datetime | None
    created_at: datetime


class MediaCounts(BaseModel):
    """Untruncated totals for the monitor/search overview."""

    image_with_crops_count: int
    person_crop_count: int


class VisualSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=20, ge=1, le=100)
    target: str = "person_crop"
    filters: SearchFilters = Field(default_factory=SearchFilters)
    rerank: bool = False


class ImageSearchRequest(BaseModel):
    image_id: uuid.UUID
    top_k: int = Field(default=20, ge=1, le=100)
    target: str = "person_crop"
    filters: SearchFilters = Field(default_factory=SearchFilters)


class SearchResultItem(BaseModel):
    crop_id: uuid.UUID | None = None
    image_id: uuid.UUID | None = None
    image_url: str | None = None
    crop_url: str | None = None
    score: float
    original_score: float | None = None
    embedding_rerank_score: float | None = None
    rerank_score: float | None = None
    rerank_reason: str | None = None
    captured_at: datetime | None = None
    location_id: uuid.UUID | None = None
    location_name: str | None = None
    camera_id: uuid.UUID | None = None
    camera_name: str | None = None
    person_id: uuid.UUID | None = None
    person_name: str | None = None
    attributes: dict[str, Any] | None = None
    labels_zh: dict[str, Any] | None = None
    labels_en: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    items: list[SearchResultItem]


class ObservationIndexItem(ORMModel):
    id: uuid.UUID
    crop_id: uuid.UUID
    image_id: uuid.UUID | None = None
    person_id: uuid.UUID | None = None
    person_name: str | None = None
    employee_no: str | None = None
    department: str | None = None
    recognition_result_type: str | None = None
    face_similarity: float | None = None
    face_confidence: float | None = None
    face_embedding_id: uuid.UUID | None = None
    face_embedding_model: str | None = None
    camera_id: uuid.UUID | None = None
    camera_name: str | None = None
    location_id: uuid.UUID | None = None
    location_name: str | None = None
    captured_at: datetime | None = None
    image_url: str | None = None
    crop_url: str | None = None
    thumbnail_url: str | None = None
    bbox: dict[str, Any] | None = None
    attributes: dict[str, Any] | None = None
    labels_zh: dict[str, Any] | None = None
    labels_en: dict[str, Any] | None = None
    search_text: str | None = None
    vl_embedding_id: uuid.UUID | None = None
    vl_embedding_model: str | None = None
    vl_embedding_dim: int | None = None
    milvus_collection: str | None = None
    milvus_object_id: str | None = None
    has_face_embedding: bool
    has_vl_embedding: bool
    created_at: datetime
    updated_at: datetime


class ObservationIndexResponse(BaseModel):
    items: list[ObservationIndexItem]
    total: int
    limit: int
    offset: int


class IndexRebuildResponse(BaseModel):
    target: str
    requested: int
    seen: int
    indexed: int
    errors: list[str] = Field(default_factory=list)


class CountingLineConfig(BaseModel):
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)


class VideoStreamCreate(BaseModel):
    name: str
    stream_url: str
    protocol: str = "rtsp"
    camera_id: uuid.UUID | None = None
    location_id: uuid.UUID | None = None
    location_name: str | None = None
    frame_interval_seconds: float = Field(default=2.0, ge=0.2, le=3600)
    reconnect_interval_seconds: int = Field(default=5, ge=1, le=300)
    counting_line: CountingLineConfig | None = None


class VideoStreamCountingLineUpdate(BaseModel):
    counting_line: CountingLineConfig | None = None


class VideoStreamRead(ORMModel):
    id: uuid.UUID
    name: str
    stream_url: str
    protocol: str
    status: str
    camera_id: uuid.UUID | None
    location_id: uuid.UUID | None
    location_name: str | None
    frame_interval_seconds: float
    reconnect_interval_seconds: int
    counting_line: CountingLineConfig | None
    last_frame_image_id: uuid.UUID | None
    last_error: str | None
    started_at: datetime | None
    stopped_at: datetime | None
    created_at: datetime
    updated_at: datetime


class StreamActionResponse(BaseModel):
    stream_id: uuid.UUID
    status: str
    message: str


class VideoProcessResponse(BaseModel):
    video_url: str
    frame_interval_seconds: float
    frames_read: int
    frames_sampled: int
    images_created: int
    crops_created: int
    counting_events_created: int
    image_ids: list[uuid.UUID]
    crop_ids: list[uuid.UUID]
