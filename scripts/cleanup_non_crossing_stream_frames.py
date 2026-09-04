from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from sqlalchemy import select, text

from app.config.settings import get_settings
from app.db.session import SessionLocal, engine
from app.models import persons  # noqa: F401
from app.models.events import CountingEvent, RecognitionEvent
from app.models.media import Image, PersonCrop
from app.models.vectors import FaceEmbedding, VLEmbedding

BACKUP_STAMP = "20260518_crossing_only_v2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove old stream frames/crops that were saved before line crossing."
    )
    parser.add_argument("--execute", action="store_true", help="delete rows and files")
    args = parser.parse_args()

    settings = get_settings()
    with SessionLocal() as db:
        counted_image_ids = {
            value
            for value in db.scalars(select(CountingEvent.image_id)).all()
            if value is not None
        }
        counted_crop_ids = {
            value
            for value in db.scalars(select(CountingEvent.crop_id)).all()
            if value is not None
        }
        counted_recognition_ids = {
            value
            for value in db.scalars(select(CountingEvent.recognition_event_id)).all()
            if value is not None
        }
        counted_crop_image_ids = {
            value
            for value in db.scalars(
                select(PersonCrop.image_id).where(PersonCrop.id.in_(counted_crop_ids))
            ).all()
            if value is not None
        }
        stream_images = list(db.scalars(select(Image).where(Image.source_type == "stream_frame")))
        stream_image_ids = {image.id for image in stream_images}
        protected_image_ids = counted_image_ids | counted_crop_image_ids
        stale_image_ids = stream_image_ids - protected_image_ids
        stream_crops = list(
            db.scalars(
                select(PersonCrop)
                .join(Image, PersonCrop.image_id == Image.id)
                .where(Image.source_type == "stream_frame")
            )
        )
        stale_crop_ids = {crop.id for crop in stream_crops if crop.id not in counted_crop_ids}
        stale_crops = [crop for crop in stream_crops if crop.id in stale_crop_ids]
        all_crop_image_ids = {
            value for value in db.scalars(select(PersonCrop.image_id)).all() if value is not None
        }
        stale_image_ids = stale_image_ids - all_crop_image_ids
        stale_images = [image for image in stream_images if image.id in stale_image_ids]
        stale_recognition_events = list(
            db.scalars(
                select(RecognitionEvent).where(
                    RecognitionEvent.id.notin_(counted_recognition_ids),
                    (RecognitionEvent.image_id.in_(stale_image_ids))
                    | (RecognitionEvent.crop_id.in_(stale_crop_ids)),
                )
            )
        )
        stale_face_embeddings = list(
            db.scalars(
                select(FaceEmbedding).where(
                    (FaceEmbedding.image_id.in_(stale_image_ids))
                    | (FaceEmbedding.crop_id.in_(stale_crop_ids))
                )
            )
        )
        stale_vl_embeddings = list(
            db.scalars(
                select(VLEmbedding).where(
                    (
                        (VLEmbedding.object_type == "image")
                        & VLEmbedding.object_id.in_(stale_image_ids)
                    )
                    | (
                        (VLEmbedding.object_type == "person_crop")
                        & VLEmbedding.object_id.in_(stale_crop_ids)
                    )
                )
            )
        )

        plan = {
            "mode": "execute" if args.execute else "dry_run",
            "stream_images_total": len(stream_images),
            "counted_images_preserved": len(protected_image_ids),
            "counted_crops_preserved": len(counted_crop_ids),
            "delete_images": len(stale_images),
            "delete_crops": len(stale_crops),
            "delete_recognition_events": len(stale_recognition_events),
            "delete_face_embeddings": len(stale_face_embeddings),
            "delete_vl_embeddings": len(stale_vl_embeddings),
        }
        print(plan)
        if not args.execute:
            return

        backup_tables()

        stale_file_rows = [(image.image_url, image.thumbnail_url) for image in stale_images]
        stale_file_rows.extend((crop.crop_url, None) for crop in stale_crops)
        delete_rows(stale_image_ids, stale_crop_ids)

        removed_files = remove_files(settings.data_dir, stale_file_rows)
        delete_milvus_entries("image", [str(value) for value in stale_image_ids])
        delete_milvus_entries("person_crop", [str(value) for value in stale_crop_ids])
        remaining = {
            "removed_files": removed_files,
            "remaining_stream_images": db.scalar(
                select(text("count(*)"))
                .select_from(Image)
                .where(Image.source_type == "stream_frame")
            ),
            "remaining_stream_crops": db.scalar(
                select(text("count(*)"))
                .select_from(PersonCrop)
                .join(Image, PersonCrop.image_id == Image.id)
                .where(Image.source_type == "stream_frame")
            ),
            "remaining_recognition_events": db.scalar(
                select(text("count(*)")).select_from(RecognitionEvent)
            ),
        }
        print(remaining)


def delete_rows(stale_image_ids: set[object], stale_crop_ids: set[object]) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                f"DELETE FROM face_embeddings WHERE id IN "
                f"(SELECT id FROM cleanup_{BACKUP_STAMP}_face_embeddings)"
            )
        )
        conn.execute(
            text(
                f"DELETE FROM vl_embeddings WHERE id IN "
                f"(SELECT id FROM cleanup_{BACKUP_STAMP}_vl_embeddings)"
            )
        )
        conn.execute(
            text(
                f"DELETE FROM recognition_events WHERE id IN "
                f"(SELECT id FROM cleanup_{BACKUP_STAMP}_recognition_events)"
            )
        )
        if stale_crop_ids:
            conn.execute(
                text("DELETE FROM person_crops WHERE id = ANY(:crop_ids)"),
                {"crop_ids": list(stale_crop_ids)},
            )
        if stale_image_ids:
            conn.execute(
                text("DELETE FROM images WHERE id = ANY(:image_ids)"),
                {"image_ids": list(stale_image_ids)},
            )


def backup_tables() -> None:
    with engine.begin() as conn:
        for table in (
            "images",
            "person_crops",
            "recognition_events",
            "face_embeddings",
            "vl_embeddings",
        ):
            conn.execute(text(f"DROP TABLE IF EXISTS cleanup_{BACKUP_STAMP}_{table}"))
            conn.execute(
                text(
                    f"CREATE TABLE cleanup_{BACKUP_STAMP}_{table} "
                    f"AS SELECT * FROM {table} WHERE false"
                )
            )
        conn.execute(
            text(
                f"INSERT INTO cleanup_{BACKUP_STAMP}_images SELECT * FROM images "
                "WHERE source_type = 'stream_frame' "
                "AND id NOT IN (SELECT image_id FROM counting_events WHERE image_id IS NOT NULL)"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO cleanup_{BACKUP_STAMP}_person_crops "
                "SELECT pc.* FROM person_crops pc "
                "JOIN images i ON pc.image_id = i.id "
                "WHERE i.source_type = 'stream_frame' "
                "AND pc.id NOT IN (SELECT crop_id FROM counting_events WHERE crop_id IS NOT NULL)"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO cleanup_{BACKUP_STAMP}_recognition_events "
                "SELECT re.* FROM recognition_events re "
                f"LEFT JOIN cleanup_{BACKUP_STAMP}_images i ON re.image_id = i.id "
                f"LEFT JOIN cleanup_{BACKUP_STAMP}_person_crops pc ON re.crop_id = pc.id "
                "WHERE (i.id IS NOT NULL OR pc.id IS NOT NULL) "
                "AND re.id NOT IN ("
                "SELECT recognition_event_id FROM counting_events "
                "WHERE recognition_event_id IS NOT NULL)"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO cleanup_{BACKUP_STAMP}_face_embeddings "
                "SELECT fe.* FROM face_embeddings fe "
                f"LEFT JOIN cleanup_{BACKUP_STAMP}_images i ON fe.image_id = i.id "
                f"LEFT JOIN cleanup_{BACKUP_STAMP}_person_crops pc ON fe.crop_id = pc.id "
                "WHERE i.id IS NOT NULL OR pc.id IS NOT NULL"
            )
        )
        conn.execute(
            text(
                f"INSERT INTO cleanup_{BACKUP_STAMP}_vl_embeddings "
                "SELECT ve.* FROM vl_embeddings ve "
                f"LEFT JOIN cleanup_{BACKUP_STAMP}_images i "
                "ON ve.object_type = 'image' AND ve.object_id = i.id "
                f"LEFT JOIN cleanup_{BACKUP_STAMP}_person_crops pc "
                "ON ve.object_type = 'person_crop' AND ve.object_id = pc.id "
                "WHERE i.id IS NOT NULL OR pc.id IS NOT NULL"
            )
        )


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
        print(
            {
                "milvus_delete": object_type,
                "ids": len(object_ids),
                "batches": deleted_batches,
            }
        )
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
