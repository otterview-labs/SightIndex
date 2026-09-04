import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class TimeRange(BaseModel):
    start_time: datetime | None = None
    end_time: datetime | None = None


class SearchFilters(TimeRange):
    camera_id: uuid.UUID | None = None
    location_id: uuid.UUID | None = None
    person_id: uuid.UUID | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
