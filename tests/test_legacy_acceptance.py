"""Server clone acceptance uses only synthetic SQLite rows and generated MJPG pixels."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from deploy.rtx5090 import legacy_acceptance as helper

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "rtx5090" / "legacy_acceptance.py"
SEED = """
import sys
from pathlib import Path
from datetime import datetime
from deploy.rtx5090.legacy_acceptance import isolated_settings
root = Path(sys.argv[1])
settings = isolated_settings(root, root/'migration.sqlite')
from app.db.session import init_db, SessionLocal, engine
from app.models.media import Image, PersonCrop, PersonObservationIndex, VideoStream
from app.models.persons import Person
from app.models.events import RecognitionEvent
from app.models.vectors import VLEmbedding, FaceEmbedding, VectorIndexJob, CropFaceExtraction
init_db()
with SessionLocal() as db:
    person = Person(name='protected-fixture-only', is_vip=True)
    image = Image(image_url='/data/frames/do-not-open.jpg', source_type='stream_frame')
    db.add_all([person, image]); db.flush()
    crop = PersonCrop(image_id=image.id, crop_url='/data/crops/do-not-open.jpg', bbox={},
                      person_id=person.id, person_id_source='manual')
    db.add(crop); db.flush()
    db.add(PersonObservationIndex(crop_id=crop.id, image_id=image.id, person_id=person.id,
                                   captured_at=datetime(2026,1,1), has_face_embedding=True,
                                   has_vl_embedding=True, person_is_vip=True))
    db.add(RecognitionEvent(crop_id=crop.id, image_id=image.id, person_id=person.id,
                             recognized_at=datetime(2026,1,2), result_type='known'))
    db.add(VLEmbedding(object_id=crop.id, object_type='reid_person_crop',
                       embedding_model='model|revision|namespace', embedding_dim=4096))
    db.add(FaceEmbedding(crop_id=crop.id, person_id=person.id,
                         face_model='fixture', embedding=[0.1,0.2]))
    db.add(VectorIndexJob(target='person_attributes', object_id=crop.id, status='running',
                          next_run_at=datetime(2026,1,3),
                          lease_owner='old-lease', lease_expires_at=None))
    db.add(VideoStream(name='fixture-camera',
                       stream_url='rtsp://do-not-open.test', status='running'))
    db.commit()
engine.dispose()
"""


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    directory = tmp_path / "acceptance"
    directory.mkdir(mode=0o700)
    env = {"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"}
    seeded = subprocess.run(
        [sys.executable, "-c", SEED, str(directory)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert seeded.returncode == 0, seeded.stderr
    directory.joinpath("migration.sqlite").chmod(0o600)
    with sqlite3.connect(directory / "migration.sqlite") as connection:
        for column in ("processed_at", "source_video_url", "video_offset_seconds"):
            connection.execute(f"ALTER TABLE images DROP COLUMN {column}")
        for column in ("absence_reason", "input_fingerprint"):
            connection.execute(f"ALTER TABLE crop_face_extractions DROP COLUMN {column}")
    return directory


def invoke(
    directory: Path, *, env: dict[str, str] | None = None, cwd: Path = ROOT
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--work-dir", str(directory)],
        cwd=cwd,
        env=env or {"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, json.loads(result.stdout)


def test_clone_migrates_twice_preserves_all_rows_and_runs_real_upload(clone: Path) -> None:
    pytest.importorskip("cv2")
    result, output = invoke(clone)
    assert result.returncode == 0, (output, result.stderr)
    assert output == {"ok": True, "migration_runs": 2, "synthetic_frames": 5, "synthetic_crops": 5}
    report = json.loads(clone.joinpath("acceptance.json").read_text())
    migration = report["migration"]
    assert migration["existing_row_counts"] == migration["after_row_counts"]
    assert migration["nullable_columns_added"] == {
        "images": ["processed_at", "source_video_url", "video_offset_seconds"],
        "crop_face_extractions": ["absence_reason", "input_fingerprint"],
    }
    assert migration["allowed_changed_columns"] == {
        "person_observation_index": ["captured_at"],
        "vector_index_jobs": ["lease_owner", "next_run_at", "status"],
    }
    assert {
        table: evidence["changed_rows"]
        for table, evidence in migration["permitted_compatibility_changes"].items()
    } == {"person_observation_index": 1, "vector_index_jobs": 1}
    for table in helper.PROTECTED_TABLES:
        assert migration["before"][table] == migration["after_existing_columns"][table]
    assert len(migration["protected_digest"]) == 64
    assert report["smoke"]["playback_checks"] == 10
    assert report["smoke"]["range_status"] == 206
    assert report["smoke"]["main_lifespan_loaded"] is False
    assert report["smoke"]["active_workers"] == 0
    assert "protected-fixture-only" not in clone.joinpath("acceptance.json").read_text()
    assert "rtsp://" not in result.stdout + result.stderr
    assert clone.joinpath("acceptance.json").stat().st_mode & 0o777 == 0o600


def test_inherited_production_env_and_dotenv_cannot_enable_workers(
    clone: Path, tmp_path: Path
) -> None:
    production = tmp_path / "production.sqlite"
    production.write_bytes(b"untouched production marker")
    original = production.read_bytes()
    clone.joinpath(".env").write_text(
        f"DATABASE_URL=sqlite:///{production}\nSTREAM_AUTOSTART_RUNNING=true\n"
        "VLM_STRUCTURED_BACKGROUND=true\nVLM_PROVIDER=openai_compatible\n"
        "APP_BASIC_AUTH_PASSWORD=never-print-dotenv-secret\n"
    )
    result, output = invoke(
        clone,
        cwd=clone,
        env={
            **os.environ,
            "DATABASE_URL": f"sqlite:///{production}",
            "MILVUS_ENABLED": "true",
            "REID_ENABLED": "true",
            "VECTOR_INDEX_ON_INGEST": "true",
            "VECTOR_INDEX_ON_INGEST_BACKGROUND": "true",
            "VLM_STRUCTURED_BACKGROUND": "true",
            "VLM_PROVIDER": "openai_compatible",
            "STREAM_AUTOSTART_RUNNING": "true",
            "FACE_RECOGNITION_ON_INGEST": "true",
            "FACE_INSIGHTFACE_ALLOW_DOWNLOAD": "true",
            "APP_BASIC_AUTH_PASSWORD": "never-print-fixture-secret",
        },
    )
    assert result.returncode == 0, (output, result.stderr)
    assert production.read_bytes() == original
    assert "never-print-fixture-secret" not in result.stdout + result.stderr
    assert "never-print-dotenv-secret" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "table,index,columns",
    [
        ("vl_embeddings", "ux_vl_embeddings_object_type_object_id", "object_type,object_id"),
        ("vector_index_jobs", "ux_vector_index_jobs_target_object", "target,object_id"),
    ],
)
def test_duplicate_markers_or_jobs_abort_before_migration(
    clone: Path,
    table: str,
    index: str,
    columns: str,
) -> None:
    with sqlite3.connect(clone / "migration.sqlite") as connection:
        # Jobs declare a UniqueConstraint (autoindex), so rebuild the synthetic table without
        # constraints rather than modifying production or hiding a duplicate from the check.
        connection.execute(f"CREATE TABLE duplicate_fixture AS SELECT * FROM {table}")
        connection.execute(f"DROP TABLE {table}")
        connection.execute(f"ALTER TABLE duplicate_fixture RENAME TO {table}")
        names = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
        projection = ["'duplicate-fixture'" if name == "id" else name for name in names]
        connection.execute(
            f"INSERT INTO {table} SELECT {','.join(projection)} FROM {table} LIMIT 1"
        )
        before = helper.snapshot(connection)
    result, output = invoke(clone)
    assert result.returncode == 1
    assert output == {"ok": False, "code": f"duplicate_{table}"}
    with sqlite3.connect(clone / "migration.sqlite") as connection:
        assert helper.snapshot(connection) == before
    assert not clone.joinpath("acceptance.json").exists()


def test_deterministic_capacity_seeds_are_allowed_existing_revisions_preserved(clone: Path) -> None:
    with sqlite3.connect(clone / "migration.sqlite") as connection:
        connection.execute(
            "UPDATE vector_index_capacity_locks SET revision=13 WHERE target='image'"
        )
        connection.execute("DELETE FROM vector_index_capacity_locks WHERE target='reid_marker'")
    result, output = invoke(clone)
    assert result.returncode == 0, (output, result.stderr)
    report = json.loads(clone.joinpath("acceptance.json").read_text())
    assert report["migration"]["capacity_targets_added"] == 1
    with sqlite3.connect(clone / "migration.sqlite") as connection:
        assert connection.execute(
            "SELECT revision FROM vector_index_capacity_locks WHERE target='image'"
        ).fetchone() == (13,)


@pytest.mark.parametrize(
    "table,column,value",
    [
        ("person_crops", "person_id_source", "'automatic'"),
        ("face_embeddings", "embedding", "'[9,9]'"),
        ("vl_embeddings", "embedding_model", "'different-index-identity'"),
        ("video_streams", "status", "'stopped'"),
        ("person_observation_index", "captured_at", "'wrong-time'"),
        ("vector_index_jobs", "status", "'failed'"),
    ],
)
def test_unexpected_data_changes_are_rejected(
    clone: Path,
    table: str,
    column: str,
    value: str,
) -> None:
    with sqlite3.connect(clone / "migration.sqlite") as connection:
        original_schema = helper.schema(connection)
        before = helper.snapshot(connection, original_schema)
        expected = helper.snapshot(connection, original_schema, expected=True)
        connection.execute(f"UPDATE {table} SET {column}={value}")
        with pytest.raises(helper.AcceptanceError, match="unexpected_existing_data_change"):
            helper.validate_migration(connection, original_schema, before, expected)


@pytest.mark.parametrize("problem", ["mode", "symlink", "hardlink", "sidecar", "report", "media"])
def test_unsafe_or_reused_work_directory_is_rejected(
    clone: Path, tmp_path: Path, problem: str
) -> None:
    target = clone
    if problem == "mode":
        clone.chmod(0o755)
    elif problem == "symlink":
        target = tmp_path / "linked"
        target.symlink_to(clone, target_is_directory=True)
    elif problem == "hardlink":
        os.link(clone / "migration.sqlite", tmp_path / "second-link.sqlite")
    elif problem == "sidecar":
        (clone / "migration.sqlite-wal").touch()
    elif problem == "report":
        (clone / "acceptance.json").touch()
    else:
        (clone / "synthetic-data").mkdir()
    result, output = invoke(target)
    assert result.returncode == 1
    assert output["ok"] is False
    assert "fixture-only" not in result.stdout + result.stderr


def test_network_or_worker_guard_has_safe_reason() -> None:
    with pytest.raises(helper.AcceptanceError, match="external_or_worker_action_attempted"):
        helper.forbidden("never-print-secret")


def test_swallowed_network_attempt_still_rejects_acceptance(clone: Path) -> None:
    code = """
import sys, socket
from deploy.rtx5090 import legacy_acceptance as helper
original = helper.synthetic_smoke
def smoke(settings):
    try:
        socket.socket().connect(('127.0.0.1', 1))
    except helper.AcceptanceError:
        pass
    return original(settings)
helper.synthetic_smoke = smoke
raise SystemExit(helper.main(['--work-dir', sys.argv[1]]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(clone)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout) == {
        "ok": False,
        "code": "external_or_worker_action_attempted",
    }
    assert not clone.joinpath("acceptance.json").exists()


@pytest.mark.parametrize(
    "table,field", [("images", "source_type"), ("person_crops", "person_id_source")]
)
def test_smoke_cannot_modify_old_media_or_manual_binding(
    clone: Path, table: str, field: str
) -> None:
    code = """
import sys, sqlite3
from deploy.rtx5090 import legacy_acceptance as helper
original = helper.synthetic_smoke
def smoke(settings):
    result = original(settings)
    with sqlite3.connect(sys.argv[1]+'/migration.sqlite') as connection:
        connection.execute('UPDATE '+sys.argv[2]+' SET '+sys.argv[3]+"='tampered' WHERE "+
                           sys.argv[3]+" IN ('stream_frame','manual')")
    return result
helper.synthetic_smoke = smoke
raise SystemExit(helper.main(['--work-dir', sys.argv[1]]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(clone), table, field],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout) == {"ok": False, "code": "smoke_changed_protected_data"}
    assert not clone.joinpath("acceptance.json").exists()


def test_nonnullable_or_backfilled_addition_is_rejected() -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE images (id TEXT)")
        connection.execute("INSERT INTO images VALUES ('fixture')")
        original_schema = helper.schema(connection)
        before = helper.snapshot(connection)
        connection.execute("ALTER TABLE images ADD COLUMN source_video_url TEXT")
        connection.execute("UPDATE images SET source_video_url='invented-source'")
        with pytest.raises(helper.AcceptanceError, match="new_column_backfilled"):
            helper.validate_migration(connection, original_schema, before, before)


def test_row_digest_detects_reassociation_even_with_same_column_multisets() -> None:
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE protected (id TEXT, person_id TEXT)")
        connection.executemany("INSERT INTO protected VALUES (?, ?)", [("a", "p"), ("b", "q")])
        before = helper.snapshot(connection)["protected"]
        connection.execute("UPDATE protected SET person_id=CASE id WHEN 'a' THEN 'q' ELSE 'p' END")
        after = helper.snapshot(connection)["protected"]
        assert before["columns"] == after["columns"]
        assert before["row_digest"] != after["row_digest"]
