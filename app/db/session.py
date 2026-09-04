from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config.settings import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
engine_kwargs: dict[str, object] = {
    "pool_pre_ping": True,
    "pool_recycle": settings.database_pool_recycle_seconds,
}
if not settings.database_url.startswith("sqlite"):
    engine_kwargs.update(
        {
            "pool_size": settings.database_pool_size,
            "max_overflow": settings.database_max_overflow,
            "pool_timeout": settings.database_pool_timeout_seconds,
        }
    )
engine = create_engine(settings.database_url, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app.models import chat, events, media, persons, vectors  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_compatible_schema()


def _ensure_compatible_schema() -> None:
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if "vector_index_capacity_locks" in table_names:
        with engine.begin() as connection:
            for target in (
                "image",
                "person_crop",
                "reid_person_crop",
                "reid_marker",
                "observation_index",
            ):
                connection.execute(
                    text(
                        "INSERT INTO vector_index_capacity_locks (target, revision) "
                        "VALUES (:target, 0) ON CONFLICT (target) DO NOTHING"
                    ),
                    {"target": target},
                )
    if "vl_embeddings" in table_names:
        with engine.begin() as connection:
            # Coverage is one logical fact per object and vector target. Older versions had no
            # database invariant, so clean any race-created duplicates before adding it.
            connection.execute(
                text(
                    "DELETE FROM vl_embeddings WHERE id IN ("
                    "SELECT id FROM ("
                    "SELECT id, ROW_NUMBER() OVER ("
                    "PARTITION BY object_type, object_id "
                    "ORDER BY created_at DESC, id DESC"
                    ") AS duplicate_rank FROM vl_embeddings"
                    ") AS ranked WHERE duplicate_rank > 1"
                    ")"
                )
            )
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "ux_vl_embeddings_object_type_object_id "
                    "ON vl_embeddings (object_type, object_id)"
                )
            )
    if "video_streams" in table_names:
        video_stream_columns = {
            column["name"] for column in inspector.get_columns("video_streams")
        }
        if "counting_line" not in video_stream_columns:
            column_type = "JSONB" if engine.dialect.name == "postgresql" else "JSON"
            with engine.begin() as connection:
                connection.execute(
                    text(f"ALTER TABLE video_streams ADD COLUMN counting_line {column_type}")
                )
    if "video_streams" in table_names:
        stream_columns = {column["name"] for column in inspector.get_columns("video_streams")}
        if "location_name" not in stream_columns:
            with engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE video_streams ADD COLUMN location_name VARCHAR")
                )

    if "person_crops" in table_names:
        person_crop_columns = {
            column["name"] for column in inspector.get_columns("person_crops")
        }
        if "attributes" not in person_crop_columns:
            column_type = "JSONB" if engine.dialect.name == "postgresql" else "JSON"
            with engine.begin() as connection:
                connection.execute(
                    text(f"ALTER TABLE person_crops ADD COLUMN attributes {column_type}")
                )
        _ensure_indexes(
            {
                "ix_person_crops_person_created": (
                    "person_crops",
                    "person_id, created_at",
                ),
                "ix_person_crops_captured_at": ("person_crops", "captured_at"),
            }
        )
    if "counting_events" in table_names:
        counting_event_columns = {
            column["name"] for column in inspector.get_columns("counting_events")
        }
        column_type = "UUID" if engine.dialect.name == "postgresql" else "CHAR(36)"
        with engine.begin() as connection:
            if "stream_id" not in counting_event_columns:
                connection.execute(
                    text(f"ALTER TABLE counting_events ADD COLUMN stream_id {column_type}")
                )
                connection.execute(
                    text("CREATE INDEX IF NOT EXISTS ix_counting_events_stream_id "
                         "ON counting_events (stream_id)")
                )
            if "image_id" not in counting_event_columns:
                connection.execute(
                    text(f"ALTER TABLE counting_events ADD COLUMN image_id {column_type}")
                )
                connection.execute(
                    text("CREATE INDEX IF NOT EXISTS ix_counting_events_image_id "
                         "ON counting_events (image_id)")
                )
            if "crop_id" not in counting_event_columns:
                connection.execute(
                    text(f"ALTER TABLE counting_events ADD COLUMN crop_id {column_type}")
                )
                connection.execute(
                    text("CREATE INDEX IF NOT EXISTS ix_counting_events_crop_id "
                         "ON counting_events (crop_id)")
                )
            _ensure_indexes(
                {
                    "ix_counting_events_crop_counted": (
                        "counting_events",
                        "crop_id, counted_at",
                    ),
                }
            )
    if "recognition_events" in table_names:
        recognition_event_columns = {
            column["name"] for column in inspector.get_columns("recognition_events")
        }
        if "face_bbox" not in recognition_event_columns:
            column_type = "JSONB" if engine.dialect.name == "postgresql" else "JSON"
            with engine.begin() as connection:
                connection.execute(
                    text(f"ALTER TABLE recognition_events ADD COLUMN face_bbox {column_type}")
                )
        _ensure_indexes(
            {
                "ix_recognition_events_person_time": (
                    "recognition_events",
                    "person_id, recognized_at",
                ),
                "ix_recognition_events_crop_created": (
                    "recognition_events",
                    "crop_id, created_at",
                ),
            }
        )
    if {
        "person_observation_index",
        "person_crops",
        "images",
        "recognition_events",
    }.issubset(table_names):
        with engine.begin() as connection:
            # Observation rows used to substitute crop.created_at when no camera timestamp
            # existed. SQLite CURRENT_TIMESTAMP is UTC-naive while captured_at is local-naive,
            # producing an irreparably mixed column. Replace only those synthetic values with
            # the latest real recognition time, or NULL when no such event exists.
            connection.execute(
                text(
                    "UPDATE person_observation_index SET captured_at=("
                    "SELECT MAX(recognized_at) FROM recognition_events "
                    "WHERE recognition_events.crop_id=person_observation_index.crop_id"
                    ") WHERE crop_id IN ("
                    "SELECT person_crops.id FROM person_crops "
                    "LEFT JOIN images ON images.id=person_crops.image_id "
                    "WHERE person_crops.captured_at IS NULL "
                    "AND (images.id IS NULL OR images.captured_at IS NULL)"
                    ")"
                )
            )
    if "vector_index_jobs" in table_names:
        vector_job_columns = {
            column["name"] for column in inspector.get_columns("vector_index_jobs")
        }
        with engine.begin() as connection:
            if "status" not in vector_job_columns:
                connection.execute(
                    text(
                        "ALTER TABLE vector_index_jobs "
                        "ADD COLUMN status VARCHAR DEFAULT 'pending' NOT NULL"
                    )
                )
            if "attempts" not in vector_job_columns:
                connection.execute(
                    text(
                        "ALTER TABLE vector_index_jobs "
                        "ADD COLUMN attempts INTEGER DEFAULT 0 NOT NULL"
                    )
                )
            if "next_run_at" not in vector_job_columns:
                connection.execute(
                    text("ALTER TABLE vector_index_jobs ADD COLUMN next_run_at TIMESTAMP")
                )
            if "last_error" not in vector_job_columns:
                connection.execute(
                    text("ALTER TABLE vector_index_jobs ADD COLUMN last_error VARCHAR")
                )
            if "lease_owner" not in vector_job_columns:
                connection.execute(
                    text("ALTER TABLE vector_index_jobs ADD COLUMN lease_owner VARCHAR")
                )
            if "lease_expires_at" not in vector_job_columns:
                connection.execute(
                    text("ALTER TABLE vector_index_jobs ADD COLUMN lease_expires_at TIMESTAMP")
                )
            # Rows claimed by the pre-lease worker have status=running but no ownership token.
            # The new claim predicate cannot recover them as expired, so normalize them before
            # dedupe (which otherwise deliberately prefers a running row over pending).
            connection.execute(
                text(
                    "UPDATE vector_index_jobs SET status='pending', next_run_at=NULL, "
                    "lease_owner=NULL, lease_expires_at=NULL "
                    "WHERE status='running' "
                    "AND (lease_owner IS NULL OR lease_expires_at IS NULL)"
                )
            )
            # One logical job per (target, object_id). Prefer a currently running row, then a
            # pending row, over a terminal failure. The set-based window query works on both
            # supported SQLite and PostgreSQL versions and avoids loading the whole queue into
            # application memory during every worker startup.
            connection.execute(
                text(
                    "DELETE FROM vector_index_jobs WHERE id IN ("
                    "SELECT id FROM ("
                    "SELECT id, ROW_NUMBER() OVER ("
                    "PARTITION BY target, object_id ORDER BY "
                    "CASE status WHEN 'running' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END, "
                    "created_at DESC, id DESC"
                    ") AS duplicate_rank FROM vector_index_jobs"
                    ") AS ranked WHERE duplicate_rank > 1"
                    ")"
                )
            )
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_vector_index_jobs_target_object "
                    "ON vector_index_jobs (target, object_id)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_vector_index_jobs_status "
                    "ON vector_index_jobs (status)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_vector_index_jobs_target "
                    "ON vector_index_jobs (target)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_vector_index_jobs_object_id "
                    "ON vector_index_jobs (object_id)"
                )
            )
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_vector_index_jobs_next_run_at "
                    "ON vector_index_jobs (next_run_at)"
                )
            )
    if "person_observation_index" in table_names:
        _ensure_indexes(
            {
                "ix_person_observation_crop": (
                    "person_observation_index",
                    "crop_id",
                ),
                "ix_person_observation_person_time": (
                    "person_observation_index",
                    "person_id, captured_at",
                ),
                "ix_person_observation_captured_at": (
                    "person_observation_index",
                    "captured_at",
                ),
                "ix_person_observation_camera_time": (
                    "person_observation_index",
                    "camera_id, captured_at",
                ),
                "ix_person_observation_location_time": (
                    "person_observation_index",
                    "location_id, captured_at",
                ),
            }
        )


def _ensure_indexes(indexes: dict[str, tuple[str, str]]) -> None:
    with engine.begin() as connection:
        for index_name, (table_name, columns) in indexes.items():
            connection.execute(
                text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns})")
            )
