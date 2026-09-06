import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base
from app.models.types import GUID


class ReidMatchFeedback(Base):
    """A human verdict for one query/candidate crop pair.

    Feedback is deliberately separate from ``PersonCrop.person_id``.  A pairwise judgement is
    enough to calibrate body, face and attribute thresholds, but it must not silently enrol a
    person or change production ranking before the labelled set has been evaluated.
    """

    __tablename__ = "reid_match_feedback"
    __table_args__ = (
        UniqueConstraint(
            "query_crop_id",
            "candidate_crop_id",
            name="ux_reid_match_feedback_pair",
        ),
        Index(
            "ix_reid_match_feedback_query_updated",
            "query_crop_id",
            "updated_at",
        ),
        Index("ix_reid_match_feedback_candidate", "candidate_crop_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    query_crop_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("person_crops.id", ondelete="CASCADE"), nullable=False
    )
    candidate_crop_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("person_crops.id", ondelete="CASCADE"), nullable=False
    )
    same_person: Mapped[bool] = mapped_column(Boolean, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="search")
    # Evidence is a snapshot of what the operator saw. The calibration script recomputes current
    # scores from the crops, while these fields explain whether a later model/config change moved
    # the result away from the state that was actually judged.
    body_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    face_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    face_reliability: Mapped[float | None] = mapped_column(Float, nullable=True)
    face_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    attribute_agreement: Mapped[float | None] = mapped_column(Float, nullable=True)
    attribute_comparable_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attribute_match_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attribute_conflict_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fusion_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_level: Mapped[str | None] = mapped_column(String(32), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
