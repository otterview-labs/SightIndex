import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base
from app.models.types import GUID, json_type


class Image(Base):
    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    image_url: Mapped[str] = mapped_column(String, nullable=False)
    thumbnail_url: Mapped[str | None] = mapped_column(String, nullable=True)
    source_type: Mapped[str] = mapped_column(String, nullable=False, default="upload")
    camera_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    location_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class PersonCrop(Base):
    __tablename__ = "person_crops"
    __table_args__ = (
        Index("ix_person_crops_person_created", "person_id", "created_at"),
        Index("ix_person_crops_captured_at", "captured_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("images.id"), nullable=False)
    crop_url: Mapped[str] = mapped_column(String, nullable=False)
    bbox: Mapped[dict[str, Any]] = mapped_column(json_type(), nullable=False)
    attributes: Mapped[dict[str, Any] | None] = mapped_column(json_type(), nullable=True)
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("persons.id"), nullable=True
    )
    camera_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    location_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class PersonObservationIndex(Base):
    __tablename__ = "person_observation_index"
    __table_args__ = (
        Index("ix_person_observation_crop", "crop_id", unique=True),
        Index("ix_person_observation_person_time", "person_id", "captured_at"),
        Index("ix_person_observation_captured_at", "captured_at"),
        Index("ix_person_observation_camera_time", "camera_id", "captured_at"),
        Index("ix_person_observation_location_time", "location_id", "captured_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    crop_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("person_crops.id"), nullable=False
    )
    image_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("images.id"), nullable=True
    )
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("persons.id"), nullable=True
    )
    person_name: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    employee_no: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    department: Mapped[str | None] = mapped_column(String, nullable=True)
    recognition_result_type: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    face_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    face_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    face_embedding_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True)
    face_embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)
    camera_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    camera_name: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    location_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    location_name: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    image_url: Mapped[str | None] = mapped_column(String, nullable=True)
    crop_url: Mapped[str | None] = mapped_column(String, nullable=True)
    thumbnail_url: Mapped[str | None] = mapped_column(String, nullable=True)
    bbox: Mapped[dict[str, Any] | None] = mapped_column(json_type(), nullable=True)
    attributes: Mapped[dict[str, Any] | None] = mapped_column(json_type(), nullable=True)
    labels_zh: Mapped[dict[str, Any] | None] = mapped_column(json_type(), nullable=True)
    labels_en: Mapped[dict[str, Any] | None] = mapped_column(json_type(), nullable=True)
    search_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    vl_embedding_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True)
    vl_embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)
    vl_embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    milvus_collection: Mapped[str | None] = mapped_column(String, nullable=True)
    milvus_object_id: Mapped[str | None] = mapped_column(String, nullable=True)
    has_face_embedding: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_vl_embedding: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class VideoStream(Base):
    __tablename__ = "video_streams"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    stream_url: Mapped[str] = mapped_column(String, nullable=False)
    protocol: Mapped[str] = mapped_column(String, nullable=False, default="rtsp")
    status: Mapped[str] = mapped_column(String, nullable=False, default="stopped", index=True)
    camera_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    location_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    # There is no Location entity; camera_name already resolves from the stream's own name, and
    # without this the observation table shows a raw UUID where the place should be.
    location_name: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    frame_interval_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=2.0)
    reconnect_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    counting_line: Mapped[dict[str, Any] | None] = mapped_column(json_type(), nullable=True)
    last_frame_image_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("images.id"), nullable=True
    )
    last_error: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
