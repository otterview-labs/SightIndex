"""Durability and lease regressions for the persistent vector-index outbox."""

import importlib
import sys
import threading
import time
import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text


def load_app(monkeypatch, tmp_path, name: str, **env: str):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / f'{name}.db'}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PERSON_DETECTOR", "whole_frame")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for module_name in list(sys.modules):
        if module_name == "main" or module_name.startswith("app."):
            sys.modules.pop(module_name)
    return importlib.import_module("main")


VL_ENV = {
    "MILVUS_ENABLED": "true",
    "VECTOR_INDEX_ON_INGEST": "true",
    "VECTOR_INDEX_ON_INGEST_BACKGROUND": "true",
    "EMBEDDING_PROVIDER": "ollama",
}


def _create_crop(SessionLocal, *, url: str = "/data/crops/q.jpg"):
    from app.models.media import Image, PersonCrop

    with SessionLocal() as db:
        image = Image(image_url="/data/frames/q.jpg", source_type="stream_frame")
        db.add(image)
        db.flush()
        crop = PersonCrop(image_id=image.id, crop_url=url, bbox={"label": "person"})
        db.add(crop)
        db.commit()
        db.refresh(crop)
        return crop.id


def _enqueue_and_claim(queue, SessionLocal, settings, target: str, object_id: uuid.UUID):
    with SessionLocal() as db:
        assert queue.enqueue_in_session(db, target, object_id, settings) is True
        db.commit()
    jobs = queue._claim_jobs(settings)
    assert len(jobs) == 1
    return jobs[0]


def test_legacy_running_job_without_lease_is_recovered_during_schema_upgrade(
    monkeypatch, tmp_path
):
    main = load_app(monkeypatch, tmp_path, "legacy-running-lease", **VL_ENV)

    from app.config.settings import get_settings
    from app.db.session import SessionLocal, engine
    from app.models.vectors import VectorIndexJob
    from app.services.vector_index_queue import VectorIndexQueue

    job_id = str(uuid.uuid4())
    object_id = str(uuid.uuid4())
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE vector_index_jobs ("
                "id CHAR(36) PRIMARY KEY, target VARCHAR NOT NULL, object_id CHAR(36) NOT NULL, "
                "status VARCHAR NOT NULL, attempts INTEGER NOT NULL, next_run_at TIMESTAMP, "
                "last_error VARCHAR, created_at TIMESTAMP NOT NULL, updated_at TIMESTAMP NOT NULL"
                ")"
            )
        )
        connection.execute(
            text(
                "INSERT INTO vector_index_jobs "
                "(id, target, object_id, status, attempts, created_at, updated_at) "
                "VALUES (:id, 'person_crop', :object_id, 'running', 1, "
                "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"id": job_id, "object_id": object_id},
        )

    main.init_db()

    columns = {column["name"] for column in inspect(engine).get_columns("vector_index_jobs")}
    assert {"lease_owner", "lease_expires_at"}.issubset(columns)
    with SessionLocal() as db:
        recovered = db.get(VectorIndexJob, uuid.UUID(job_id))
        assert recovered is not None
        assert recovered.status == "pending"
        assert recovered.lease_owner is None
        assert recovered.lease_expires_at is None

    claims = VectorIndexQueue()._claim_jobs(get_settings())
    assert [claim.id for claim in claims] == [uuid.UUID(job_id)]


def test_legacy_duplicate_vector_markers_are_deduplicated_before_unique_index(
    monkeypatch, tmp_path
):
    main = load_app(monkeypatch, tmp_path, "legacy-vector-marker-duplicates", **VL_ENV)

    from app.db.session import engine

    object_id = str(uuid.uuid4())
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE vl_embeddings ("
                "id CHAR(36) PRIMARY KEY, object_type VARCHAR NOT NULL, "
                "object_id CHAR(36) NOT NULL, embedding JSON, "
                "embedding_model VARCHAR NOT NULL, embedding_dim INTEGER NOT NULL, "
                "created_at TIMESTAMP NOT NULL"
                ")"
            )
        )
        for marker_id in (str(uuid.uuid4()), str(uuid.uuid4())):
            connection.execute(
                text(
                    "INSERT INTO vl_embeddings "
                    "(id, object_type, object_id, embedding_model, embedding_dim, created_at) "
                    "VALUES (:id, 'reid_person_crop', :object_id, 'legacy', 4096, "
                    "CURRENT_TIMESTAMP)"
                ),
                {"id": marker_id, "object_id": object_id},
            )

    main.init_db()

    with engine.connect() as connection:
        remaining = connection.scalar(
            text(
                "SELECT COUNT(*) FROM vl_embeddings "
                "WHERE object_type='reid_person_crop' AND object_id=:object_id"
            ),
            {"object_id": object_id},
        )
    indexes = {index["name"]: index for index in inspect(engine).get_indexes("vl_embeddings")}
    assert remaining == 1
    assert indexes["ux_vl_embeddings_object_type_object_id"]["unique"] == 1


def test_vl_worker_flushes_then_commits_marker_and_ack(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "vl-durable-success", **VL_ENV)

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.vectors import VectorIndexJob, VLEmbedding
    from app.services.vector_index import MilvusVectorIndex, VectorIndexingService
    from app.services.vector_index_queue import VectorIndexQueue

    with TestClient(main.create_app()):
        pass
    settings = get_settings()
    crop_id = _create_crop(SessionLocal)
    queue = VectorIndexQueue()
    job = _enqueue_and_claim(queue, SessionLocal, settings, "person_crop", crop_id)
    calls: list[str] = []

    monkeypatch.setattr(
        VectorIndexingService,
        "write_crop_vector",
        lambda self, crop, flush=False: calls.append("write"),
    )
    monkeypatch.setattr(
        MilvusVectorIndex,
        "flush",
        lambda self, target: calls.append("flush"),
    )

    def record(self, crop):
        calls.append("record")
        self.db.add(
            VLEmbedding(
                object_type="person_crop",
                object_id=crop.id,
                embedding=None,
                embedding_model="test",
                embedding_dim=8,
            )
        )

    monkeypatch.setattr(VectorIndexingService, "record_crop_index", record)
    queue._run_vl_jobs([job], settings)

    with SessionLocal() as db:
        assert db.query(VectorIndexJob).count() == 0
        assert db.query(VLEmbedding).filter_by(object_id=crop_id).count() == 1
    assert calls == ["write", "flush", "record"]


def test_vl_flush_failure_keeps_retryable_job_without_marker(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "vl-flush-failure", **VL_ENV)

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.vectors import VectorIndexJob, VLEmbedding
    from app.services.vector_index import (
        MilvusVectorIndex,
        VectorIndexError,
        VectorIndexingService,
    )
    from app.services.vector_index_queue import VectorIndexQueue

    with TestClient(main.create_app()):
        pass
    settings = get_settings()
    crop_id = _create_crop(SessionLocal)
    queue = VectorIndexQueue()
    job = _enqueue_and_claim(queue, SessionLocal, settings, "person_crop", crop_id)

    monkeypatch.setattr(
        VectorIndexingService,
        "write_crop_vector",
        lambda self, crop, flush=False: None,
    )
    monkeypatch.setattr(
        MilvusVectorIndex,
        "flush",
        lambda self, target: (_ for _ in ()).throw(VectorIndexError("flush boom")),
    )
    monkeypatch.setattr(
        VectorIndexingService,
        "record_crop_index",
        lambda self, crop: pytest.fail("marker must not be recorded before a durable flush"),
    )

    queue._run_vl_jobs([job], settings)

    with SessionLocal() as db:
        remaining = db.query(VectorIndexJob).one()
        assert remaining.status == "pending"
        assert remaining.attempts == 1
        assert "flush boom" in remaining.last_error
        assert db.query(VLEmbedding).count() == 0


def test_milvus_cooldown_cannot_fake_vl_queue_success(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "vl-cooldown-fail-closed", **VL_ENV)

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.vectors import VectorIndexJob, VLEmbedding
    from app.services.vector_index import MilvusVectorIndex
    from app.services.vector_index_queue import VectorIndexQueue

    with TestClient(main.create_app()):
        pass
    settings = get_settings()
    crop_id = _create_crop(SessionLocal, url="/data/crops/missing.jpg")
    queue = VectorIndexQueue()
    job = _enqueue_and_claim(queue, SessionLocal, settings, "person_crop", crop_id)
    monkeypatch.setattr(
        MilvusVectorIndex,
        "_is_in_failure_cooldown",
        lambda self: True,
    )

    queue._run_vl_jobs([job], settings)

    with SessionLocal() as db:
        remaining = db.query(VectorIndexJob).one()
        assert remaining.status == "pending"
        assert remaining.attempts == 1
        assert "failure cooldown" in remaining.last_error
        assert db.query(VLEmbedding).count() == 0


def test_atomic_outbox_rolls_back_crop_and_all_targets_when_one_target_is_full(
    monkeypatch, tmp_path
):
    main = load_app(
        monkeypatch,
        tmp_path,
        "outbox-capacity",
        **VL_ENV,
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        REID_INDEX_ON_INGEST="true",
        VECTOR_INDEX_BACKGROUND_MAX_QUEUE="1",
    )

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.models.vectors import VectorIndexJob
    from app.services.reid_index import REID_OBJECT_TYPE
    from app.services.vector_index_queue import VectorIndexQueue, VectorQueueFullError

    with TestClient(main.create_app()):
        pass
    settings = get_settings()
    queue = VectorIndexQueue()
    with SessionLocal() as db:
        db.add(
            VectorIndexJob(
                target="person_crop",
                object_id=uuid.uuid4(),
                status="pending",
            )
        )
        db.commit()

    with SessionLocal() as db:
        image = Image(image_url="/data/frames/full.jpg", source_type="stream_frame")
        db.add(image)
        db.flush()
        crop = PersonCrop(image_id=image.id, crop_url="/data/crops/full.jpg", bbox={})
        db.add(crop)
        db.flush()
        crop_id = crop.id
        with pytest.raises(VectorQueueFullError):
            queue.enqueue_many_in_session(
                db,
                [(REID_OBJECT_TYPE, crop_id), ("person_crop", crop_id)],
                settings,
            )
        db.rollback()

    with SessionLocal() as db:
        assert db.get(PersonCrop, crop_id) is None
        assert db.query(VectorIndexJob).filter_by(object_id=crop_id).count() == 0


def test_capacity_reservation_serializes_two_transactions(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "capacity-serialization",
        **VL_ENV,
        VECTOR_INDEX_BACKGROUND_MAX_QUEUE="1",
    )

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.vectors import VectorIndexJob
    from app.services.vector_index_queue import VectorIndexQueue, VectorQueueFullError

    with TestClient(main.create_app()):
        pass
    settings = get_settings()
    queue = VectorIndexQueue()
    attempting = threading.Event()
    outcome: list[str] = []

    with SessionLocal() as first:
        assert queue.enqueue_in_session(
            first,
            "person_crop",
            uuid.uuid4(),
            settings,
        )

        def reserve_last_slot() -> None:
            with SessionLocal() as second:
                attempting.set()
                try:
                    queue.enqueue_in_session(
                        second,
                        "person_crop",
                        uuid.uuid4(),
                        settings,
                    )
                    second.commit()
                    outcome.append("admitted")
                except VectorQueueFullError:
                    second.rollback()
                    outcome.append("full")

        contender = threading.Thread(target=reserve_last_slot)
        contender.start()
        assert attempting.wait(1.0)
        time.sleep(0.1)
        first.commit()
        contender.join(timeout=3.0)

    assert contender.is_alive() is False
    assert outcome == ["full"]
    with SessionLocal() as db:
        assert db.query(VectorIndexJob).filter_by(status="pending").count() == 1


def test_process_endpoint_reports_full_outbox_and_does_not_commit_crop(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "outbox-capacity-api",
        **VL_ENV,
        VECTOR_INDEX_BACKGROUND_MAX_QUEUE="1",
    )

    from app.db.session import SessionLocal
    from app.models.media import PersonCrop
    from app.models.vectors import VectorIndexJob
    from app.services.vector_index_queue import vector_index_queue

    monkeypatch.setattr(vector_index_queue, "start", lambda settings=None: None)
    with TestClient(main.create_app()) as client:
        with SessionLocal() as db:
            db.add(
                VectorIndexJob(
                    target="person_crop",
                    object_id=uuid.uuid4(),
                    status="pending",
                )
            )
            db.commit()
        upload = client.post(
            "/api/images/upload",
            files={"file": ("frame.jpg", b"\xff\xd8\xff\xdbfake", "image/jpeg")},
        )
        assert upload.status_code == 200
        response = client.post(f"/api/images/{upload.json()['id']}/process")

    assert response.status_code == 503
    assert "queue target=person_crop" in response.json()["detail"]
    with SessionLocal() as db:
        assert db.query(PersonCrop).count() == 0
    assert list((tmp_path / "data" / "crops").glob("*")) == []


def test_image_upload_reports_full_outbox_and_removes_upload(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "image-outbox-capacity-api",
        **VL_ENV,
        VECTOR_INDEX_BACKGROUND_MAX_QUEUE="1",
    )

    from app.db.session import SessionLocal
    from app.models.media import Image
    from app.models.vectors import VectorIndexJob
    from app.services.vector_index_queue import vector_index_queue

    monkeypatch.setattr(vector_index_queue, "start", lambda settings=None: None)
    with TestClient(main.create_app()) as client:
        with SessionLocal() as db:
            db.add(
                VectorIndexJob(
                    target="image",
                    object_id=uuid.uuid4(),
                    status="pending",
                )
            )
            db.commit()
        response = client.post(
            "/api/images/upload",
            files={"file": ("frame.jpg", b"image", "image/jpeg")},
        )

    assert response.status_code == 503
    assert "queue target=image" in response.json()["detail"]
    with SessionLocal() as db:
        assert db.query(Image).count() == 0
        assert db.query(VectorIndexJob).filter_by(target="image").count() == 1
    assert list((tmp_path / "data" / "uploads").glob("*")) == []


def test_synchronous_ingest_commits_image_and_crop_markers(monkeypatch, tmp_path):
    sync_env = {
        **VL_ENV,
        "VECTOR_INDEX_ON_INGEST_BACKGROUND": "false",
    }
    main = load_app(monkeypatch, tmp_path, "sync-marker-commit", **sync_env)

    from app.db.session import SessionLocal
    from app.models.vectors import VectorIndexJob, VLEmbedding
    from app.services.vector_index import MilvusVectorIndex, VectorIndexingService

    calls: list[str] = []
    monkeypatch.setattr(
        VectorIndexingService,
        "write_image_vector",
        lambda self, image, flush=False: calls.append("write:image"),
    )
    monkeypatch.setattr(
        VectorIndexingService,
        "write_crop_vector",
        lambda self, crop, flush=False: calls.append("write:person_crop"),
    )
    monkeypatch.setattr(
        MilvusVectorIndex,
        "flush",
        lambda self, target: calls.append(f"flush:{target}"),
    )

    with TestClient(main.create_app()) as client:
        upload = client.post(
            "/api/images/upload",
            files={"file": ("frame.jpg", b"image", "image/jpeg")},
        )
        assert upload.status_code == 200
        process = client.post(f"/api/images/{upload.json()['id']}/process")
        assert process.status_code == 200
        crop_id = process.json()[0]["id"]

    with SessionLocal() as db:
        markers = {
            (marker.object_type, str(marker.object_id))
            for marker in db.query(VLEmbedding).all()
        }
        assert db.query(VectorIndexJob).count() == 0

    assert ("image", upload.json()["id"]) in markers
    assert ("person_crop", crop_id) in markers
    assert calls == [
        "write:image",
        "flush:image",
        "write:person_crop",
        "flush:person_crop",
    ]


def test_observation_upsert_is_serialized_across_workers(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "observation-upsert-serialization")

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop, PersonObservationIndex
    from app.services.observation_index import ObservationIndexService

    with TestClient(main.create_app()):
        pass
    with SessionLocal() as db:
        image = Image(image_url="/data/frames/o.jpg", source_type="stream_frame")
        db.add(image)
        db.flush()
        crop = PersonCrop(image_id=image.id, crop_url="/data/crops/o.jpg", bbox={})
        db.add(crop)
        db.commit()
        crop_id = crop.id

    start = threading.Barrier(2)
    errors: list[Exception] = []

    def upsert() -> None:
        try:
            with SessionLocal() as db:
                crop = db.get(PersonCrop, crop_id)
                assert crop is not None
                start.wait(timeout=3)
                ObservationIndexService(db, get_settings()).upsert_crop(crop)
                db.commit()
        except Exception as exc:
            errors.append(exc)

    workers = [threading.Thread(target=upsert) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)

    assert all(worker.is_alive() is False for worker in workers)
    assert errors == []
    with SessionLocal() as db:
        assert (
            db.query(PersonObservationIndex)
            .filter(PersonObservationIndex.crop_id == crop_id)
            .count()
            == 1
        )


def test_stale_owner_cannot_ack_after_expired_job_is_reclaimed(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "lease-fencing", **VL_ENV)

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.vectors import VectorIndexJob
    from app.services.time_utils import local_now
    from app.services.vector_index_queue import LeaseLostError, VectorIndexQueue

    with TestClient(main.create_app()):
        pass
    settings = get_settings()
    crop_id = _create_crop(SessionLocal)
    worker_a = VectorIndexQueue()
    stale_claim = _enqueue_and_claim(
        worker_a, SessionLocal, settings, "person_crop", crop_id
    )
    with SessionLocal() as db:
        row = db.get(VectorIndexJob, stale_claim.id)
        row.lease_expires_at = local_now(settings) - timedelta(seconds=1)
        db.add(row)
        db.commit()

    worker_b = VectorIndexQueue()
    fresh_claims = worker_b._claim_jobs(settings)
    assert len(fresh_claims) == 1
    assert fresh_claims[0].lease_owner != stale_claim.lease_owner

    with pytest.raises(LeaseLostError):
        worker_a._commit_vl_success(stale_claim, settings, object_exists=False)
    with SessionLocal() as db:
        row = db.get(VectorIndexJob, stale_claim.id)
        assert row is not None
        assert row.lease_owner == fresh_claims[0].lease_owner


def test_stale_owner_cannot_requeue_after_expired_job_is_reclaimed(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "failure-fencing", **VL_ENV)

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.vectors import VectorIndexJob
    from app.services.time_utils import local_now
    from app.services.vector_index_queue import VectorIndexQueue

    with TestClient(main.create_app()):
        pass
    settings = get_settings()
    crop_id = _create_crop(SessionLocal)
    worker_a = VectorIndexQueue()
    stale_claim = _enqueue_and_claim(
        worker_a, SessionLocal, settings, "person_crop", crop_id
    )
    with SessionLocal() as db:
        row = db.get(VectorIndexJob, stale_claim.id)
        assert row is not None
        row.lease_expires_at = local_now(settings) - timedelta(seconds=1)
        db.commit()

    worker_b = VectorIndexQueue()
    fresh_claims = worker_b._claim_jobs(settings)
    assert len(fresh_claims) == 1
    fresh_claim = fresh_claims[0]
    assert fresh_claim.lease_owner != stale_claim.lease_owner

    worker_a._mark_failed(stale_claim, "late failure", settings)

    with SessionLocal() as db:
        row = db.get(VectorIndexJob, stale_claim.id)
        assert row is not None
        assert row.status == "running"
        assert row.lease_owner == fresh_claim.lease_owner
        assert row.attempts == fresh_claim.attempts
        assert row.last_error is None


def test_lease_heartbeat_renews_and_fences_before_commit(monkeypatch, tmp_path):
    load_app(monkeypatch, tmp_path, "lease-heartbeat", **VL_ENV)

    from app.config.settings import get_settings
    from app.services.vector_index_queue import (
        ClaimedVectorIndexJob,
        VectorIndexQueue,
        _LeaseHeartbeat,
    )

    queue = VectorIndexQueue()
    settings = get_settings().model_copy(update={"vector_index_lease_seconds": 0.3})
    claim = ClaimedVectorIndexJob(
        id=uuid.uuid4(),
        target="person_crop",
        object_id=uuid.uuid4(),
        attempts=0,
        lease_owner="worker-a",
    )
    calls: list[tuple[uuid.UUID, ...]] = []

    def renew(job_ids, owner, current_settings):
        calls.append(job_ids)
        return True

    monkeypatch.setattr(queue, "_renew_leases", renew)
    with _LeaseHeartbeat(queue, (claim,), settings) as heartbeat:
        time.sleep(0.65)
        heartbeat.fence()
    assert len(calls) >= 3  # enter, heartbeat, final fence


def test_stop_timeout_keeps_live_thread_and_prevents_second_worker(monkeypatch, tmp_path):
    load_app(monkeypatch, tmp_path, "stop-timeout", **VL_ENV)

    from app.config.settings import get_settings
    from app.services.vector_index_queue import VectorIndexQueue

    queue = VectorIndexQueue()
    release = threading.Event()
    thread = threading.Thread(target=lambda: release.wait(2.0), daemon=True)
    thread.start()
    queue._thread = thread

    queue.stop(timeout=0.01)
    assert queue._thread is thread
    queue.start(get_settings())
    assert queue._thread is thread

    release.set()
    thread.join(timeout=1.0)
    queue.stop(timeout=0.1)
    assert queue._thread is None


def test_reid_split_rethrows_non_input_error(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "split-policy",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import PersonCrop
    from app.services.reid import ReidEmbeddingService, ReidRuntimeError
    from app.services.reid_index import ReidIndexService

    with TestClient(main.create_app()):
        pass
    crops_dir = tmp_path / "data" / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    (crops_dir / "split.jpg").write_bytes(b"img")
    crop_id = _create_crop(SessionLocal, url="/data/crops/split.jpg")
    monkeypatch.setattr(
        ReidEmbeddingService,
        "embed_images",
        lambda self, paths: (_ for _ in ()).throw(
            ReidRuntimeError("bad batch", status_code=422)
        ),
    )
    monkeypatch.setattr(
        ReidEmbeddingService,
        "embed_image",
        lambda self, path: (_ for _ in ()).throw(
            ReidRuntimeError("rate limited", status_code=429)
        ),
    )

    with SessionLocal() as db:
        crop = db.get(PersonCrop, crop_id)
        assert crop is not None
        with pytest.raises(ReidRuntimeError, match="rate limited") as exc_info:
            ReidIndexService(db, get_settings()).index_crops_batch([crop])
    assert exc_info.value.status_code == 429
