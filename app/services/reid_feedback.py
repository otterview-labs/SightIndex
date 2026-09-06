import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.media import PersonCrop
from app.models.reid import ReidMatchFeedback
from app.schemas.reid import ReidFeedbackUpsert


class FeedbackCropNotFoundError(LookupError):
    def __init__(self, role: str, crop_id: uuid.UUID) -> None:
        self.role = role
        self.crop_id = crop_id
        super().__init__(f"{role} crop not found: {crop_id}")


class ReidFeedbackService:
    """Stores calibration labels without feeding them back into live ranking."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def upsert(self, payload: ReidFeedbackUpsert) -> ReidMatchFeedback:
        if payload.query_crop_id == payload.candidate_crop_id:
            raise ValueError("Query crop and candidate crop must be different")
        self._require_crops(payload.query_crop_id, payload.candidate_crop_id)

        feedback = self.db.scalar(
            select(ReidMatchFeedback).where(
                ReidMatchFeedback.query_crop_id == payload.query_crop_id,
                ReidMatchFeedback.candidate_crop_id == payload.candidate_crop_id,
            )
        )
        if feedback is None:
            feedback = ReidMatchFeedback(**payload.model_dump())
        else:
            for field, value in payload.model_dump(
                exclude={"query_crop_id", "candidate_crop_id"}
            ).items():
                setattr(feedback, field, value)
        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)
        return feedback

    def list_for_query(
        self,
        query_crop_id: uuid.UUID,
        *,
        limit: int = 500,
    ) -> list[ReidMatchFeedback]:
        self._require_crop("Query", query_crop_id)
        return list(
            self.db.scalars(
                select(ReidMatchFeedback)
                .where(ReidMatchFeedback.query_crop_id == query_crop_id)
                .order_by(ReidMatchFeedback.updated_at.desc())
                .limit(limit)
            )
        )

    def export_all(self) -> list[ReidMatchFeedback]:
        return list(
            self.db.scalars(
                select(ReidMatchFeedback).order_by(
                    ReidMatchFeedback.created_at.asc(),
                    ReidMatchFeedback.id.asc(),
                )
            )
        )

    def _require_crops(
        self,
        query_crop_id: uuid.UUID,
        candidate_crop_id: uuid.UUID,
    ) -> None:
        crop_ids = set(
            self.db.scalars(
                select(PersonCrop.id).where(
                    PersonCrop.id.in_([query_crop_id, candidate_crop_id])
                )
            )
        )
        if query_crop_id not in crop_ids:
            raise FeedbackCropNotFoundError("Query", query_crop_id)
        if candidate_crop_id not in crop_ids:
            raise FeedbackCropNotFoundError("Candidate", candidate_crop_id)

    def _require_crop(self, role: str, crop_id: uuid.UUID) -> None:
        if self.db.get(PersonCrop, crop_id) is None:
            raise FeedbackCropNotFoundError(role, crop_id)
