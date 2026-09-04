from datetime import datetime

from fastapi import APIRouter

from app.api.deps import DBSession
from app.schemas.events import CountSummary
from app.services.statistics import StatisticsService

router = APIRouter(prefix="/statistics", tags=["statistics"])


@router.get("/count", response_model=CountSummary)
def count_events(
    db: DBSession,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> CountSummary:
    return StatisticsService(db).count_summary(start_time=start_time, end_time=end_time)


@router.get("/persons")
def count_persons(db: DBSession) -> dict[str, int]:
    summary = StatisticsService(db).count_summary()
    return {"unique_person_count": summary.unique_person_count}


@router.get("/unknown")
def count_unknown(db: DBSession) -> dict[str, int]:
    summary = StatisticsService(db).count_summary()
    return {"unique_unknown_count": summary.unique_unknown_count}


@router.get("/by-location")
def count_by_location() -> dict[str, list[object]]:
    return {"items": []}


@router.get("/by-day")
def count_by_day() -> dict[str, list[object]]:
    return {"items": []}
