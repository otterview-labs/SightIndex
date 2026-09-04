import uuid
from datetime import datetime

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app.models.events import CountingEvent, RecognitionEvent
from app.models.media import Image, PersonCrop
from app.schemas.events import CountSummary


class StatisticsService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def count_summary(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        stream_id: uuid.UUID | None = None,
    ) -> CountSummary:
        recognition_stmt = select(func.count()).select_from(RecognitionEvent)
        counting_stmt = select(func.count()).select_from(CountingEvent)
        image_stmt = select(func.count()).select_from(Image)
        person_crop_stmt = select(func.count()).select_from(PersonCrop)
        unique_person_stmt = select(func.count(distinct(RecognitionEvent.person_id))).where(
            RecognitionEvent.person_id.is_not(None)
        )
        unique_unknown_stmt = select(
            func.count(distinct(RecognitionEvent.unknown_cluster_id))
        ).where(RecognitionEvent.unknown_cluster_id.is_not(None))
        if start_time:
            recognition_stmt = recognition_stmt.where(RecognitionEvent.recognized_at >= start_time)
            counting_stmt = counting_stmt.where(CountingEvent.counted_at >= start_time)
            image_stmt = image_stmt.where(Image.created_at >= start_time)
            person_crop_stmt = person_crop_stmt.where(PersonCrop.created_at >= start_time)
            unique_person_stmt = unique_person_stmt.where(
                RecognitionEvent.recognized_at >= start_time
            )
            unique_unknown_stmt = unique_unknown_stmt.where(
                RecognitionEvent.recognized_at >= start_time
            )
        if end_time:
            recognition_stmt = recognition_stmt.where(RecognitionEvent.recognized_at <= end_time)
            counting_stmt = counting_stmt.where(CountingEvent.counted_at <= end_time)
            image_stmt = image_stmt.where(Image.created_at <= end_time)
            person_crop_stmt = person_crop_stmt.where(PersonCrop.created_at <= end_time)
            unique_person_stmt = unique_person_stmt.where(
                RecognitionEvent.recognized_at <= end_time
            )
            unique_unknown_stmt = unique_unknown_stmt.where(
                RecognitionEvent.recognized_at <= end_time
            )
        if stream_id:
            counting_stmt = counting_stmt.where(CountingEvent.stream_id == stream_id)
        return CountSummary(
            recognition_event_count=self.db.scalar(recognition_stmt) or 0,
            counting_event_count=self.db.scalar(counting_stmt) or 0,
            unique_person_count=self.db.scalar(unique_person_stmt) or 0,
            unique_unknown_count=self.db.scalar(unique_unknown_stmt) or 0,
            image_count=self.db.scalar(image_stmt) or 0,
            person_crop_count=self.db.scalar(person_crop_stmt) or 0,
        )
