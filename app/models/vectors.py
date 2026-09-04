import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base
from app.models.types import GUID, json_type


class VLEmbedding(Base):
    __tablename__ = "vl_embeddings"
    __table_args__ = (
        Index(
            "ux_vl_embeddings_object_type_object_id",
            "object_type",
            "object_id",
            unique=True,
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    object_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    object_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    embedding: Mapped[list[float] | None] = mapped_column(json_type(), nullable=True)
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class VectorIndexJob(Base):
    __tablename__ = "vector_index_jobs"
    __table_args__ = (
        UniqueConstraint(
            "target",
            "object_id",
            name="ux_vector_index_jobs_target_object",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    target: Mapped[str] = mapped_column(String, nullable=False, index=True)
    object_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    lease_owner: Mapped[str | None] = mapped_column(String, nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class VectorIndexCapacityLock(Base):
    """One database row per target used to serialize capacity reservations."""

    __tablename__ = "vector_index_capacity_locks"

    target: Mapped[str] = mapped_column(String, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    person_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("persons.id"), nullable=True
    )
    image_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("images.id"), nullable=True
    )
    crop_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(), ForeignKey("person_crops.id"), nullable=True
    )
    face_bbox: Mapped[dict[str, Any] | None] = mapped_column(json_type(), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(json_type(), nullable=True)
    face_model: Mapped[str] = mapped_column(String, nullable=False)
    quality_score: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
