import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base
from app.models.types import GUID


class RecognitionEvent(Base):
    __tablename__ = "recognition_events"
    __table_args__ = (
        Index("ix_recognition_events_person_time", "person_id", "recognized_at"),
        Index("ix_recognition_events_crop_created", "crop_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("images.id"), nullable=True
    )
    crop_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("person_crops.id"), nullable=True
    )
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("persons.id"), nullable=True
    )
    unknown_cluster_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    camera_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    location_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    confidence: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    similarity: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    face_bbox: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    result_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    recognized_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CountingEvent(Base):
    __tablename__ = "counting_events"
    __table_args__ = (
        Index("ix_counting_events_crop_counted", "crop_id", "counted_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    stream_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("video_streams.id"), nullable=True, index=True
    )
    image_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("images.id"), nullable=True
    )
    crop_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("person_crops.id"), nullable=True
    )
    recognition_event_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("recognition_events.id"), nullable=True
    )
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("persons.id"), nullable=True
    )
    unknown_cluster_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    camera_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    location_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    count_type: Mapped[str] = mapped_column(String, nullable=False, default="appearance")
    direction: Mapped[str | None] = mapped_column(String, nullable=True)
    counted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
