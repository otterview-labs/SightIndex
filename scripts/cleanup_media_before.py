from __future__ import annotations

# ruff: noqa: E402 -- CLI bootstrap adds the repository root before importing app modules.
import argparse
import sys
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import or_, select, text

from app.config.settings import get_settings
from app.db.session import SessionLocal, engine
from app.models import persons  # noqa: F401
from app.models.events import CountingEvent, RecognitionEvent
from app.models.media import Image, PersonCrop, PersonObservationIndex
from app.models.reid import ReidMatchFeedback
from app.models.vectors import CropFaceExtraction, FaceEmbedding, VectorIndexJob, VLEmbedding

# Tables that must be cleared before person_crops or images. reid_match_feedback declares
# ON DELETE CASCADE; every other reference is NO ACTION and rejects the parent delete outright.
# Which of them carry a constraint varies by how the database was created, so all of them are
# cleared unconditionally. Keep this list in step with the schema.
CHILD_TABLES_BEFORE_CROPS = (
    "reid_match_feedback",
    "person_observation_index",
    "crop_face_extractions",
    "vector_index_jobs",
    "counting_events",
    "recognition_events",
    "face_embeddings",
    "vl_embeddings",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove media data before a cutoff time.")
    cutoff_group = parser.add_mutually_exclusive_group()
    cutoff_group.add_argument(
        "--before",
        help="Cutoff datetime, for example 2026-05-18T00:00:00+08:00",
    )
    cutoff_group.add_argument(
        "--retention-days",
        type=int,
        help="Keep this many days of media. Defaults to MEDIA_RETENTION_DAYS.",
    )
    parser.add_argument("--execute", action="store_true", help="delete rows and files")
    args = parser.parse_args()

    settings = get_settings()
    cutoff = retention_cutoff(settings, args.before, args.retention_days)
    backup_stamp = "cleanup_before_" + cutoff.strftime("%Y%m%d_%H%M%S")

    with SessionLocal() as db:
        images = list(
            db.scalars(
                select(Image).where(
                    or_(
                        Image.created_at < cutoff,
                        Image.captured_at < cutoff,
                    )
                )
            )
        )
        image_ids = {image.id for image in images}

        crops = list(
            db.scalars(
                select(PersonCrop).where(
                    or_(
                        PersonCrop.created_at < cutoff,
                        PersonCrop.captured_at < cutoff,
                        PersonCrop.image_id.in_(image_ids),
                    )
                )
            )
        )
        crop_ids = {crop.id for crop in crops}

        recognition_events = list(
            db.scalars(
                select(RecognitionEvent).where(
                    or_(
                        RecognitionEvent.created_at < cutoff,
                        RecognitionEvent.recognized_at < cutoff,
                        RecognitionEvent.image_id.in_(image_ids),
                        RecognitionEvent.crop_id.in_(crop_ids),
                    )
                )
            )
        )
        recognition_ids = {event.id for event in recognition_events}

        counting_events = list(
            db.scalars(
                select(CountingEvent).where(
                    or_(
                        CountingEvent.created_at < cutoff,
                        CountingEvent.counted_at < cutoff,
                        CountingEvent.image_id.in_(image_ids),
                        CountingEvent.crop_id.in_(crop_ids),
                        CountingEvent.recognition_event_id.in_(recognition_ids),
                    )
                )
            )
        )
        counting_ids = {event.id for event in counting_events}

        face_embeddings = list(
            db.scalars(
                select(FaceEmbedding).where(
                    or_(
                        FaceEmbedding.image_id.in_(image_ids),
                        FaceEmbedding.crop_id.in_(crop_ids),
                    )
                )
            )
        )
        face_embedding_ids = {embedding.id for embedding in face_embeddings}

        vl_embeddings = list(
            db.scalars(
                select(VLEmbedding).where(
                    or_(
                        (
                            (VLEmbedding.object_type == "image")
                            & VLEmbedding.object_id.in_(image_ids)
                        ),
                        (
                            (VLEmbedding.object_type == "person_crop")
                            & VLEmbedding.object_id.in_(crop_ids)
                        ),
                        VLEmbedding.created_at < cutoff,
                    )
                )
            )
        )
        vl_embedding_ids = {embedding.id for embedding in vl_embeddings}

        observation_rows = list(
            db.scalars(
                select(PersonObservationIndex).where(
                    or_(
                        PersonObservationIndex.crop_id.in_(crop_ids),
                        PersonObservationIndex.image_id.in_(image_ids),
                    )
                )
            )
        )
        observation_ids = {row.id for row in observation_rows}

        # This table is keyed by crop_id; it has no id column of its own.
        face_cache_crop_ids = {
            crop_id
            for crop_id in db.scalars(
                select(CropFaceExtraction.crop_id).where(
                    CropFaceExtraction.crop_id.in_(crop_ids)
                )
            )
        }

        feedback_rows = list(
            db.scalars(
                select(ReidMatchFeedback).where(
                    or_(
                        ReidMatchFeedback.query_crop_id.in_(crop_ids),
                        ReidMatchFeedback.candidate_crop_id.in_(crop_ids),
                    )
                )
            )
        )
        feedback_ids = {row.id for row in feedback_rows}

        # Durable queue rows: leaving them behind lets the worker retry object_ids that no
        # longer exist, and re-create vectors for deleted crops.
        referenced_object_ids = image_ids | crop_ids
        index_job_rows = list(
            db.scalars(
                select(VectorIndexJob).where(VectorIndexJob.object_id.in_(referenced_object_ids))
            )
        )
        index_job_ids = {job.id for job in index_job_rows}

        referenced_image_ids = {
            value
            for value in db.scalars(
                select(PersonCrop.image_id).where(PersonCrop.id.notin_(crop_ids))
            ).all()
            if value is not None
        }
        image_ids_to_delete = image_ids - referenced_image_ids
        images_to_delete = [image for image in images if image.id in image_ids_to_delete]

        file_rows = [(image.image_url, image.thumbnail_url) for image in images_to_delete]
        file_rows.extend((crop.crop_url, None) for crop in crops)

        plan = {
            "mode": "execute" if args.execute else "dry_run",
            "cutoff": cutoff.isoformat(),
            "backup_stamp": backup_stamp,
            "delete_images": len(images_to_delete),
            "preserve_images_still_referenced": len(image_ids - image_ids_to_delete),
            "delete_crops": len(crops),
            "delete_counting_events": len(counting_events),
            "delete_recognition_events": len(recognition_events),
            "delete_face_embeddings": len(face_embeddings),
            "delete_vl_embeddings": len(vl_embeddings),
            "delete_person_observation_index": len(observation_rows),
            "delete_crop_face_extractions": len(face_cache_crop_ids),
            "delete_reid_match_feedback": len(feedback_rows),
            "delete_vector_index_jobs": len(index_job_rows),
        }
        print(plan)
        if not args.execute:
            return

        backup_tables(
            backup_stamp=backup_stamp,
            image_ids=image_ids_to_delete,
            crop_ids=crop_ids,
            counting_ids=counting_ids,
            recognition_ids=recognition_ids,
            face_embedding_ids=face_embedding_ids,
            vl_embedding_ids=vl_embedding_ids,
            observation_ids=observation_ids,
            face_cache_crop_ids=face_cache_crop_ids,
            feedback_ids=feedback_ids,
            index_job_ids=index_job_ids,
        )
        delete_rows(
            image_ids=image_ids_to_delete,
            crop_ids=crop_ids,
            counting_ids=counting_ids,
            recognition_ids=recognition_ids,
            face_embedding_ids=face_embedding_ids,
            vl_embedding_ids=vl_embedding_ids,
            observation_ids=observation_ids,
            face_cache_crop_ids=face_cache_crop_ids,
            feedback_ids=feedback_ids,
            index_job_ids=index_job_ids,
        )
        removed_files = remove_files(settings.data_dir, file_rows)
        delete_milvus_entries("image", [str(value) for value in image_ids_to_delete])
        delete_milvus_entries("person_crop", [str(value) for value in crop_ids])
        # ReID keeps its own collection: a different model and dimension from the VL vectors.
        delete_milvus_entries("reid_person_crop", [str(value) for value in crop_ids])
        print(
            {
                "removed_files": removed_files,
                "remaining_images": db.scalar(select(text("count(*)")).select_from(Image)),
                "remaining_crops": db.scalar(select(text("count(*)")).select_from(PersonCrop)),
                "remaining_observations": db.scalar(
                    select(text("count(*)")).select_from(PersonObservationIndex)
                ),
                "remaining_crop_face_extractions": db.scalar(
                    select(text("count(*)")).select_from(CropFaceExtraction)
                ),
                "remaining_counting_events": db.scalar(
                    select(text("count(*)")).select_from(CountingEvent)
                ),
                "remaining_recognition_events": db.scalar(
                    select(text("count(*)")).select_from(RecognitionEvent)
                ),
                "remaining_vector_index_jobs": db.scalar(
                    select(text("count(*)")).select_from(VectorIndexJob)
                ),
            }
        )


def parse_cutoff(value: str, timezone_name: str) -> datetime:
    cutoff = datetime.fromisoformat(value)
    if cutoff.tzinfo is None:
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            timezone = ZoneInfo("Asia/Shanghai")
        cutoff = cutoff.replace(tzinfo=timezone)
    return cutoff


def retention_cutoff(settings: object, before: str | None, retention_days: int | None) -> datetime:
    if before:
        return parse_cutoff(before, settings.local_timezone)
    days = retention_days or settings.media_retention_days
    try:
        timezone = ZoneInfo(settings.local_timezone)
    except ZoneInfoNotFoundError:
        timezone = ZoneInfo("Asia/Shanghai")
    return datetime.now(timezone) - timedelta(days=days)


def backup_tables(
    *,
    backup_stamp: str,
    image_ids: set[object],
    crop_ids: set[object],
    counting_ids: set[object],
    recognition_ids: set[object],
    face_embedding_ids: set[object],
    vl_embedding_ids: set[object],
    observation_ids: set[object],
    face_cache_crop_ids: set[object],
    feedback_ids: set[object],
    index_job_ids: set[object],
) -> None:
    with engine.begin() as conn:
        for table in ("images", "person_crops", *CHILD_TABLES_BEFORE_CROPS):
            conn.execute(text(f"DROP TABLE IF EXISTS {backup_stamp}_{table}"))
            conn.execute(
                text(f"CREATE TABLE {backup_stamp}_{table} AS SELECT * FROM {table} WHERE false")
            )
        insert_backup(conn, backup_stamp, "images", image_ids)
        insert_backup(conn, backup_stamp, "person_crops", crop_ids)
        insert_backup(conn, backup_stamp, "counting_events", counting_ids)
        insert_backup(conn, backup_stamp, "recognition_events", recognition_ids)
        insert_backup(conn, backup_stamp, "face_embeddings", face_embedding_ids)
        insert_backup(conn, backup_stamp, "vl_embeddings", vl_embedding_ids)
        insert_backup(conn, backup_stamp, "person_observation_index", observation_ids)
        insert_backup(
            conn, backup_stamp, "crop_face_extractions", face_cache_crop_ids, key="crop_id"
        )
        insert_backup(conn, backup_stamp, "reid_match_feedback", feedback_ids)
        insert_backup(conn, backup_stamp, "vector_index_jobs", index_job_ids)


def insert_backup(
    conn: object,
    backup_stamp: str,
    table: str,
    ids: set[object],
    key: str = "id",
) -> None:
    if not ids:
        return
    conn.execute(
        text(f"INSERT INTO {backup_stamp}_{table} SELECT * FROM {table} WHERE {key} = ANY(:ids)"),
        {"ids": list(ids)},
    )


def delete_rows(
    *,
    image_ids: set[object],
    crop_ids: set[object],
    counting_ids: set[object],
    recognition_ids: set[object],
    face_embedding_ids: set[object],
    vl_embedding_ids: set[object],
    observation_ids: set[object],
    face_cache_crop_ids: set[object],
    feedback_ids: set[object],
    index_job_ids: set[object],
) -> None:
    with engine.begin() as conn:
        if image_ids:
            conn.execute(
                text(
                    "UPDATE video_streams SET last_frame_image_id = NULL "
                    "WHERE last_frame_image_id = ANY(:ids)"
                ),
                {"ids": list(image_ids)},
            )
        # Order matters here. PostgreSQL checks a NO ACTION reference when the statement ends.
        # The deployed schema (milvus_log.test) carries such references from
        # person_observation_index and recognition_events onto person_crops and images, so those
        # rows have to go first or the person_crops delete is rejected. crop_face_extractions,
        # counting_events and face_embeddings only carry the constraint in databases created
        # from a newer model set; clearing them unconditionally keeps this order valid either
        # way. reid_match_feedback would cascade, but deleting it explicitly preserves the audit
        # trail in the backup tables instead of losing it to the cascade.
        delete_by_ids(conn, "reid_match_feedback", feedback_ids)
        delete_by_ids(conn, "person_observation_index", observation_ids)
        delete_by_ids(conn, "vector_index_jobs", index_job_ids)
        if face_cache_crop_ids:
            conn.execute(
                text("DELETE FROM crop_face_extractions WHERE crop_id = ANY(:ids)"),
                {"ids": list(face_cache_crop_ids)},
            )
        delete_by_ids(conn, "counting_events", counting_ids)
        delete_by_ids(conn, "face_embeddings", face_embedding_ids)
        delete_by_ids(conn, "vl_embeddings", vl_embedding_ids)
        delete_by_ids(conn, "recognition_events", recognition_ids)
        delete_by_ids(conn, "person_crops", crop_ids)
        delete_by_ids(conn, "images", image_ids)


def delete_by_ids(conn: object, table: str, ids: set[object]) -> None:
    if not ids:
        return
    conn.execute(text(f"DELETE FROM {table} WHERE id = ANY(:ids)"), {"ids": list(ids)})


def delete_milvus_entries(object_type: str, object_ids: list[str]) -> None:
    if not object_ids:
        return
    settings = get_settings()
    if not settings.milvus_enabled:
        return
    try:
        from app.services.vector_index import MilvusVectorIndex
    except Exception as exc:
        print({"milvus_delete": object_type, "status": "skipped", "error": str(exc)})
        return
    try:
        collection = MilvusVectorIndex(settings)._collection(object_type)
        deleted_batches = 0
        for batch in batched(object_ids, 200):
            quoted = ", ".join(f'"{object_id}"' for object_id in batch)
            collection.delete(expr=f"object_id in [{quoted}]")
            deleted_batches += 1
        collection.flush()
        print({"milvus_delete": object_type, "ids": len(object_ids), "batches": deleted_batches})
    except Exception as exc:
        print({"milvus_delete": object_type, "status": "failed", "error": str(exc)})


def batched(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def remove_files(data_dir: Path, urls: list[tuple[str | None, str | None]]) -> int:
    removed = 0
    for row in urls:
        for url in row:
            if not url or not str(url).startswith("/data/"):
                continue
            path = data_dir / str(url).removeprefix("/data/")
            try:
                if path.exists():
                    path.unlink()
                    removed += 1
            except OSError as exc:
                print({"file_remove_error": str(path), "error": str(exc)})
    return removed


if __name__ == "__main__":
    main()
