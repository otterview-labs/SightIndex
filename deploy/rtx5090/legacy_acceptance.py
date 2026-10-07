"""Offline, clone-only acceptance for an existing SQLite deployment.

Run with the existing virtualenv from the staged code checkout. The operator must create
``migration.sqlite`` using SQLite's backup API inside a fresh, owner-only work directory.
This command never opens a configurable production database, starts the app lifespan,
loads inference weights, or reads business media. All printed evidence is aggregate-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import socket
import sqlite3
import stat
import sys
import threading
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch


class AcceptanceError(RuntimeError):
    """Contains a safe reason code, never a database value or exception body."""


NULLABLE_ADDITIONS = {
    "images": {"processed_at", "source_video_url", "video_offset_seconds"},
    "crop_face_extractions": {"absence_reason", "input_fingerprint"},
    "video_streams": {"counting_line", "last_frame_read_at", "location_name"},
    "person_crops": {"person_id_source", "attributes"},
    "counting_events": {"stream_id", "image_id", "crop_id"},
    "recognition_events": {"face_bbox"},
    "vector_index_jobs": {"next_run_at", "last_error", "lease_owner", "lease_expires_at"},
}
PROTECTED_TABLES = {
    "persons",
    "person_crops",
    "face_embeddings",
    "crop_face_extractions",
    "vl_embeddings",
    "recognition_events",
    "reid_match_feedback",
    "video_streams",
}
CAPACITY_TARGETS = (
    "image",
    "person_crop",
    "reid_person_crop",
    "reid_marker",
    "observation_index",
    "person_attributes",
)


def require(condition: bool, code: str) -> None:
    if not condition:
        raise AcceptanceError(code)


def validate_work_dir(raw_path: Path) -> tuple[Path, Path]:
    require(raw_path.is_absolute(), "work_dir_not_absolute")
    require(".." not in raw_path.parts, "unsafe_work_dir")
    for component in (raw_path, *raw_path.parents):
        require(not component.is_symlink(), "work_dir_symlink")
    directory = raw_path.stat()
    require(stat.S_ISDIR(directory.st_mode), "work_dir_not_directory")
    require(stat.S_IMODE(directory.st_mode) == 0o700, "work_dir_not_private")
    require(directory.st_uid == os.geteuid(), "work_dir_wrong_owner")
    database = raw_path / "migration.sqlite"
    info = database.lstat()
    require(stat.S_ISREG(info.st_mode), "clone_not_regular")
    require(info.st_nlink == 1, "clone_multiple_links")
    require(info.st_uid == os.geteuid(), "clone_wrong_owner")
    require(stat.S_IMODE(info.st_mode) == 0o600, "clone_not_private")
    require(info.st_size > 0, "clone_empty")
    require(not (raw_path / "acceptance.json").exists(), "report_already_exists")
    require(not (raw_path / "synthetic-data").exists(), "synthetic_data_already_exists")
    for suffix in ("-wal", "-shm", "-journal"):
        require(not Path(f"{database}{suffix}").exists(), "clone_sidecar_present")
    return raw_path, database


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def connect_readonly(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def schema(connection: sqlite3.Connection) -> dict[str, list[str]]:
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    return {
        name: [row[1] for row in connection.execute(f"PRAGMA table_info({quote(name)})")]
        for (name,) in tables
    }


def check_duplicates(connection: sqlite3.Connection, tables: dict[str, list[str]]) -> None:
    for table, columns in (
        ("vl_embeddings", ("object_type", "object_id")),
        ("vector_index_jobs", ("target", "object_id")),
    ):
        if table not in tables:
            continue
        require(set(columns) <= set(tables[table]), "duplicate_check_schema_missing")
        keys = ", ".join(quote(column) for column in columns)
        duplicate = connection.execute(
            f"SELECT 1 FROM {quote(table)} GROUP BY {keys} HAVING COUNT(*)>1 LIMIT 1"
        ).fetchone()
        require(duplicate is None, f"duplicate_{table}")


def cell_digest(value: object) -> str:
    if value is None:
        data = b"null"
    elif isinstance(value, bytes):
        data = b"bytes:" + value
    else:
        data = (type(value).__name__ + ":" + repr(value)).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def digest_list(values: list[str]) -> str:
    return hashlib.sha256("".join(sorted(values)).encode("ascii")).hexdigest()


def expected_expression(table: str, column: str, tables: dict[str, list[str]]) -> str:
    plain = f"original.{quote(column)}"
    if table == "person_observation_index" and column == "captured_at":
        required = {"person_observation_index", "person_crops", "images", "recognition_events"}
        if required <= tables.keys():
            return (
                "CASE WHEN original.crop_id IN (SELECT person_crops.id FROM person_crops "
                "LEFT JOIN images ON images.id=person_crops.image_id "
                "WHERE person_crops.captured_at IS NULL "
                "AND (images.id IS NULL OR images.captured_at IS NULL)) THEN "
                "(SELECT MAX(recognized_at) FROM recognition_events "
                "WHERE recognition_events.crop_id=original.crop_id) ELSE original.captured_at END"
            )
    if table == "vector_index_jobs" and "status" in tables[table]:
        owner = "original.lease_owner" if "lease_owner" in tables[table] else "NULL"
        expires = "original.lease_expires_at" if "lease_expires_at" in tables[table] else "NULL"
        condition = f"original.status='running' AND ({owner} IS NULL OR {expires} IS NULL)"
        if column == "status":
            return f"CASE WHEN {condition} THEN 'pending' ELSE {plain} END"
        if column in {"next_run_at", "lease_owner", "lease_expires_at"}:
            return f"CASE WHEN {condition} THEN NULL ELSE {plain} END"
    return plain


def snapshot(
    connection: sqlite3.Connection,
    tables: dict[str, list[str]] | None = None,
    *,
    expected: bool = False,
    exclude_ids: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    tables = schema(connection) if tables is None else tables
    result: dict[str, dict[str, Any]] = {}
    for table, columns in tables.items():
        expressions = [
            expected_expression(table, column, tables) if expected else f"original.{quote(column)}"
            for column in columns
        ]
        column_hashes: list[list[str]] = [[] for _ in columns]
        row_hashes: list[str] = []
        excluded = (exclude_ids or {}).get(table, [])
        predicate = (
            f" WHERE original.id NOT IN ({','.join('?' for _ in excluded)})" if excluded else ""
        )
        rows = connection.execute(
            f"SELECT {', '.join(expressions)} FROM {quote(table)} AS original{predicate}",
            excluded,
        )
        existing_targets: set[str] = set()
        for row in rows:
            if table == "vector_index_capacity_locks":
                require(set(columns) == {"target", "revision"}, "capacity_schema_unknown")
                existing_targets.add(row[columns.index("target")])
            cells = [cell_digest(value) for value in row]
            for hashes, cell in zip(column_hashes, cells, strict=True):
                hashes.append(cell)
            row_hashes.append(hashlib.sha256("".join(cells).encode("ascii")).hexdigest())
        if expected and table == "vector_index_capacity_locks":
            for target in set(CAPACITY_TARGETS) - existing_targets:
                cells = [cell_digest(target if column == "target" else 0) for column in columns]
                for hashes, cell in zip(column_hashes, cells, strict=True):
                    hashes.append(cell)
                row_hashes.append(hashlib.sha256("".join(cells).encode("ascii")).hexdigest())
        result[table] = {
            "rows": len(row_hashes),
            "row_digest": digest_list(row_hashes),
            "columns": {
                column: digest_list(hashes)
                for column, hashes in zip(columns, column_hashes, strict=True)
            },
        }
    return result


def permitted_transform_evidence(
    connection: sqlite3.Connection, tables: dict[str, list[str]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for table, column, reason in (
        (
            "person_observation_index",
            "captured_at",
            "Replace synthetic capture times only when both source capture times are absent; "
            "use the latest real recognition time or NULL.",
        ),
        (
            "vector_index_jobs",
            "status",
            "Return only running jobs missing an owner or lease expiry to pending; "
            "clear their lease and next-run fields, retaining IDs, attempts and errors.",
        ),
    ):
        if table not in tables or column not in tables[table]:
            continue
        expression = expected_expression(table, column, tables)
        changed = connection.execute(
            f"SELECT COUNT(*) FROM {quote(table)} AS original "
            f"WHERE original.{quote(column)} IS NOT ({expression})"
        ).fetchone()[0]
        result[table] = {"changed_rows": changed, "reason": reason}
    return result


def validate_migration(
    connection: sqlite3.Connection,
    before_schema: dict[str, list[str]],
    before: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    after_schema = schema(connection)
    require(before_schema.keys() <= after_schema.keys(), "existing_table_removed")
    require(
        all(set(columns) <= set(after_schema[table]) for table, columns in before_schema.items()),
        "existing_column_removed",
    )
    after = snapshot(connection, before_schema)
    require(after == expected, "unexpected_existing_data_change")
    added: dict[str, list[str]] = {}
    for table, columns in before_schema.items():
        additions = set(after_schema[table]) - set(columns)
        require(additions <= NULLABLE_ADDITIONS.get(table, set()), "unapproved_column_added")
        for column in additions:
            info = next(
                row
                for row in connection.execute(f"PRAGMA table_info({quote(table)})")
                if row[1] == column
            )
            require(not info[3] and info[4] is None, "new_column_not_nullable")
            nonnull = connection.execute(
                f"SELECT 1 FROM {quote(table)} WHERE {quote(column)} IS NOT NULL LIMIT 1"
            ).fetchone()
            require(nonnull is None, "new_column_backfilled")
        if additions:
            added[table] = sorted(additions)
    for table in set(after_schema) - set(before_schema):
        count = connection.execute(f"SELECT COUNT(*) FROM {quote(table)}").fetchone()[0]
        if table == "vector_index_capacity_locks":
            rows = connection.execute(f"SELECT target, revision FROM {quote(table)}").fetchall()
            require(
                set(rows) == {(target, 0) for target in CAPACITY_TARGETS},
                "unexpected_new_capacity_locks",
            )
        else:
            require(count == 0, "unexpected_new_table_data")
    changes = {
        table: sorted(
            column
            for column in before[table]["columns"]
            if before[table]["columns"][column] != after[table]["columns"][column]
        )
        for table in before
        if before[table] != after[table]
    }
    protected = {table: after[table] for table in sorted(PROTECTED_TABLES & after.keys())}
    return {
        "existing_tables": len(before),
        "existing_row_counts": {table: item["rows"] for table, item in before.items()},
        "after_row_counts": {table: item["rows"] for table, item in after.items()},
        "capacity_targets_added": (
            after["vector_index_capacity_locks"]["rows"]
            - before["vector_index_capacity_locks"]["rows"]
            if "vector_index_capacity_locks" in before
            else len(CAPACITY_TARGETS)
        ),
        "before": before,
        "after_existing_columns": after,
        "nullable_columns_added": added,
        "allowed_changed_columns": changes,
        "protected_digest": hashlib.sha256(
            json.dumps(protected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def isolated_settings(work_dir: Path, database: Path) -> Any:
    """Called in a fresh process, before any database or service imports."""

    require(
        not any(name == "main" or name.startswith("app.") for name in sys.modules),
        "app_preimported",
    )
    source_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.find_spec("app")
    require(
        spec is not None
        and spec.origin is not None
        and Path(spec.origin).resolve() == source_root / "app" / "__init__.py",
        "app_source_root_wrong",
    )
    os.environ.clear()
    os.environ.update({"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"})
    import app.config.settings as settings_module

    values: dict[str, Any] = {
        "database_url": f"sqlite:///{database}",
        "data_dir": work_dir / "synthetic-data",
        "auto_create_tables": False,
        "person_detector": "whole_frame",
        "person_crop_dedupe_enabled": False,
        "person_crop_min_confidence": 0.0,
        "person_crop_min_bbox_width": 0,
        "person_crop_min_bbox_height": 0,
        "person_crop_require_whole_body": False,
        "person_crop_upscale_min_width": 0,
        "person_crop_upscale_min_height": 0,
        "person_crop_sharpen_amount": 0.0,
        "face_embedding_provider": "none",
        "face_insightface_allow_download": False,
        "face_recognition_on_ingest": False,
        "face_library_cache_enabled": False,
        "appearance_tone_on_ingest": False,
        "stream_autostart_running": False,
        "stream_diagnostics_enabled": False,
        "stream_store_empty_frames": False,
        "milvus_enabled": False,
        "embedding_provider": "none",
        "visual_embedding_provider": "none",
        "semantic_search_enabled": False,
        "vlm_provider": "none",
        "vlm_rerank_provider": "none",
        "vlm_caption_on_index": False,
        "vlm_structured_on_ingest": False,
        "vlm_structured_background": False,
        "vlm_rerank_enabled": False,
        "embedding_rerank_enabled": False,
        "reid_enabled": False,
        "reid_index_on_ingest": False,
        "reid_face_priority_enabled": False,
        "vector_index_on_ingest": False,
        "vector_index_on_ingest_background": False,
        "person_trajectory_vector_enabled": False,
        "app_basic_auth_username": None,
        "app_basic_auth_password": None,
    }
    settings = settings_module.Settings(_env_file=None, **values)
    settings_module.get_settings = lambda: settings
    return settings


def forbidden(*_args: Any, **_kwargs: Any) -> None:
    raise AcceptanceError("external_or_worker_action_attempted")


@contextmanager
def isolation_guards() -> Iterator[None]:
    from app.services.stream_runtime import StreamRuntime
    from app.services.vector_index_queue import VectorIndexQueue

    attempts: list[bool] = []

    def guarded_forbidden(*args: Any, **kwargs: Any) -> None:
        attempts.append(True)
        forbidden(*args, **kwargs)

    with ExitStack() as stack:
        for target in (
            (socket.socket, "connect"),
            (socket.socket, "connect_ex"),
            (StreamRuntime, "start"),
            (VectorIndexQueue, "start"),
            (VectorIndexQueue, "wake"),
        ):
            stack.enter_context(patch.object(*target, guarded_forbidden))
        yield
        require(not attempts, "external_or_worker_action_attempted")


def synthetic_smoke(settings: Any) -> tuple[dict[str, Any], dict[str, list[str]]]:
    import cv2
    import numpy as np
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.media import router
    from app.api.media_files import MediaStaticFiles
    from app.config.settings import get_settings
    from app.db.session import SessionLocal, get_db
    from app.models.media import Image, PersonCrop
    from app.services.storage import StorageService
    from app.services.time_utils import database_datetime

    StorageService(settings).ensure_dirs()
    fixture = settings.data_dir / "fixture.avi"
    writer = cv2.VideoWriter(str(fixture), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    require(writer.isOpened(), "synthetic_encoder_unavailable")
    try:
        for index in range(5):
            writer.write(np.full((48, 64, 3), 40 + index * 30, dtype=np.uint8))
    finally:
        writer.release()
    contents = fixture.read_bytes()
    base_captured_at = datetime(2026, 10, 7, tzinfo=UTC)
    application = FastAPI()
    application.include_router(router, prefix="/api")
    application.dependency_overrides[get_settings] = lambda: settings

    def clone_db() -> Iterator[Any]:
        with SessionLocal() as db:
            yield db

    application.dependency_overrides[get_db] = clone_db
    application.mount("/data", MediaStaticFiles(directory=settings.data_dir), name="data")
    with TestClient(application) as client:
        upload = client.post(
            "/api/videos/upload",
            files={"file": ("fixture.avi", contents, "video/x-msvideo")},
            params={
                "captured_at": base_captured_at.isoformat(),
                "frame_interval_seconds": 0.1,
                "max_frames": 5,
            },
        )
        require(upload.status_code == 200, "synthetic_upload_failed")
        result = upload.json()
        require(
            all(
                result.get(key) == 5
                for key in (
                    "frames_read",
                    "frames_sampled",
                    "images_created",
                    "crops_created",
                )
            ),
            "synthetic_counts_wrong",
        )
        images, crops = result["image_ids"], result["crop_ids"]
        require(
            len(images) == len(set(images)) == len(crops) == len(set(crops)) == 5,
            "synthetic_ids_wrong",
        )
        source = result["video_url"]
        for index, (image_id, crop_id) in enumerate(zip(images, crops, strict=True)):
            with SessionLocal() as db:
                image = db.get(Image, uuid.UUID(image_id))
                crop = db.get(PersonCrop, uuid.UUID(crop_id))
                require(image is not None and crop is not None, "synthetic_rows_missing")
                require(crop.image_id == image.id, "synthetic_crop_wrong_parent")
                require(image.source_video_url == source, "synthetic_provenance_wrong")
                require(
                    image.video_offset_seconds is not None
                    and math.isclose(image.video_offset_seconds, index / 10.0, abs_tol=0.005),
                    "synthetic_offset_wrong",
                )
                require(image.processed_at is not None, "synthetic_frame_not_processed")
                require(
                    image.captured_at
                    == database_datetime(base_captured_at, settings, "sqlite")
                    + timedelta(seconds=index / 10.0),
                    "synthetic_capture_time_wrong",
                )
                for url in (image.image_url, crop.crop_url):
                    require(
                        cv2.imread(str(settings.data_dir / url.removeprefix("/data/"))) is not None,
                        "synthetic_pixels_unreadable",
                    )
            for route in (f"images/{image_id}", f"person-crops/{crop_id}"):
                response = client.get(f"/api/{route}/playback")
                require(response.status_code == 200, "synthetic_playback_failed")
                evidence = response.json()
                require(
                    evidence["available"]
                    and evidence["video_url"] == source
                    and math.isclose(evidence["offset_seconds"], index / 10.0, abs_tol=0.005),
                    "synthetic_playback_wrong_parent",
                )
        response = client.get(source, headers={"Range": "bytes=0-15"})
        require(
            response.status_code == 206 and response.content == contents[:16],
            "synthetic_range_failed",
        )
        for target in ("images", "person-crops"):
            require(
                client.get(f"/api/{target}/{uuid.uuid4()}/playback").status_code == 404,
                "synthetic_missing_id_associated",
            )
        with SessionLocal() as db:
            legacy = Image(image_url="/data/frames/legacy.jpg", source_type="video_frame")
            still = Image(image_url="/data/frames/still.jpg", source_type="upload")
            orphan = PersonCrop(image_id=uuid.uuid4(), crop_url="/data/crops/orphan.jpg", bbox={})
            db.add_all([legacy, still, orphan])
            db.commit()
            negative_ids = [(str(legacy.id), "source_missing"), (str(still.id), "not_video")]
            orphan_id = str(orphan.id)
        for image_id, reason in negative_ids:
            response = client.get(f"/api/images/{image_id}/playback")
            require(
                response.status_code == 200
                and response.json()["reason"] == reason
                and not response.json()["available"],
                "synthetic_unknown_source_associated",
            )
        require(
            client.get(f"/api/person-crops/{orphan_id}/playback").status_code == 404,
            "synthetic_orphan_associated",
        )
    require("main" not in sys.modules, "main_imported")
    require(
        not any(thread.name.startswith("sightindex-") for thread in threading.enumerate()),
        "active_worker_detected",
    )
    report = {
        "frames": 5,
        "crops": 5,
        "playback_checks": 10,
        "offsets_seconds": [0.0, 0.1, 0.2, 0.3, 0.4],
        "range_status": 206,
        "unknown_ids_status": 404,
        "main_lifespan_loaded": False,
        "active_workers": 0,
        "negative_association_checks": 5,
    }
    # GUID stores canonical UUID text on SQLite. Only these exact freshly returned IDs
    # can be excluded from the post-smoke protection digest; never scan for a URL prefix.
    return report, {
        "images": [str(uuid.UUID(item)) for item in images + [pair[0] for pair in negative_ids]],
        "person_crops": [str(uuid.UUID(item)) for item in crops + [orphan_id]],
    }


def run(work_dir: Path) -> dict[str, Any]:
    directory, database = validate_work_dir(work_dir)
    settings = isolated_settings(directory, database)
    from app.db import session

    try:
        with connect_readonly(database) as connection:
            require(
                connection.execute("PRAGMA quick_check").fetchone() == ("ok",),
                "clone_integrity_failed",
            )
            original_schema = schema(connection)
            check_duplicates(connection, original_schema)
            before = snapshot(connection, original_schema)
            expected = snapshot(connection, original_schema, expected=True)
            compatible_changes = permitted_transform_evidence(connection, original_schema)
        with isolation_guards():
            session.init_db()
            with connect_readonly(database) as connection:
                migration = validate_migration(connection, original_schema, before, expected)
                migration["permitted_compatibility_changes"] = compatible_changes
                first = snapshot(connection)
            session.init_db()
            with connect_readonly(database) as connection:
                require(snapshot(connection) == first, "second_migration_not_idempotent")
                require(
                    connection.execute("PRAGMA quick_check").fetchone() == ("ok",),
                    "migrated_clone_integrity_failed",
                )
            smoke, synthetic_ids = synthetic_smoke(settings)
            with connect_readonly(database) as connection:
                require(
                    snapshot(connection, exclude_ids=synthetic_ids) == first,
                    "smoke_changed_protected_data",
                )
            app_root = Path(__file__).resolve().parents[2] / "app"
            require(
                all(
                    Path(module.__file__).resolve().is_relative_to(app_root)
                    for name, module in sys.modules.items()
                    if name.startswith("app.") and getattr(module, "__file__", None)
                ),
                "app_source_root_changed",
            )
        return {"ok": True, "migration": migration, "migration_runs": 2, "smoke": smoke}
    finally:
        session.engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        report = run(args.work_dir)
        target = args.work_dir / "acceptance.json"
        with target.open("x", encoding="utf-8") as output:
            json.dump(report, output, sort_keys=True, indent=2)
            output.write("\n")
        print(
            json.dumps(
                {"ok": True, "migration_runs": 2, "synthetic_frames": 5, "synthetic_crops": 5}
            )
        )
        return 0
    except Exception as exc:
        code = str(exc) if isinstance(exc, AcceptanceError) else "acceptance_internal_failure"
        print(json.dumps({"ok": False, "code": code}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
