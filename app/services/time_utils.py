from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config.settings import Settings


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
