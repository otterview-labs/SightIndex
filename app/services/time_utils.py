from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config.settings import Settings
from app.schemas.common import SearchFilters


def local_timezone(settings: Settings) -> ZoneInfo:
    try:
        return ZoneInfo(settings.local_timezone)
    except ZoneInfoNotFoundError:
        return ZoneInfo("Asia/Shanghai")


def local_now(settings: Settings) -> datetime:
    return datetime.now(local_timezone(settings))


def database_datetime(value: datetime, settings: Settings, dialect_name: str) -> datetime:
    """Normalize an API/runtime timestamp to the database's storage semantics.

    PostgreSQL preserves timezone-aware values. SQLite's DateTime adapter drops offsets without
    conversion, so convert to the configured local wall clock first; all captured_at values then
    share one unambiguous naive convention on SQLite.
    """

    timezone = local_timezone(settings)
    aware_value = value.replace(tzinfo=timezone) if value.tzinfo is None else value
    if dialect_name == "sqlite":
        return aware_value.astimezone(timezone).replace(tzinfo=None)
    return aware_value


def database_search_filters(
    filters: SearchFilters, settings: Settings, dialect_name: str
) -> SearchFilters:
    start = (
        database_datetime(filters.start_time, settings, dialect_name)
        if filters.start_time is not None
        else None
    )
    end = (
        database_datetime(filters.end_time, settings, dialect_name)
        if filters.end_time is not None
        else None
    )
    if start is not None and end is not None and start > end:
        raise ValueError("开始时间不能晚于结束时间")
    return filters.model_copy(update={"start_time": start, "end_time": end})
