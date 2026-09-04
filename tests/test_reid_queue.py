"""Ingest-time ReID indexing: gating, batching, retry bookkeeping, model switches."""

import importlib
import sys
import threading

from fastapi.testclient import TestClient


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


def test_reid_ingest_enqueues_with_vl_indexing_off(monkeypatch, tmp_path):
    """ReID ingest must run on its own flags; VL indexing being off cannot suppress it."""

    main = load_app(
        monkeypatch,
        tmp_path,
        "test-reid-enqueue",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        REID_INDEX_ON_INGEST="true",
        MILVUS_ENABLED="true",
        VECTOR_INDEX_ON_INGEST="false",
    )

    from app.db.session import SessionLocal
    from app.models.vectors import VectorIndexJob
    from app.services.vector_index_queue import vector_index_queue

    monkeypatch.setattr(vector_index_queue, "start", lambda settings=None: None)

    with TestClient(main.create_app()) as client:
        upload = client.post(
            "/api/images/upload",
            files={"file": ("frame.jpg", b"\xff\xd8\xff\xdbfake", "image/jpeg")},
        )
        assert upload.status_code == 200
        process = client.post(f"/api/images/{upload.json()['id']}/process")
        assert process.status_code == 200
        crop_id = process.json()[0]["id"]

        with SessionLocal() as db:
            jobs = list(db.query(VectorIndexJob).all())
        # Exactly one job, for the reid target: the VL target stayed off.
        assert [job.target for job in jobs] == ["reid_person_crop"]
        assert str(jobs[0].object_id) == crop_id

        # Enqueueing the same crop again must not duplicate the job.
        vector_index_queue.enqueue_reid_crop(jobs[0].object_id)
        with SessionLocal() as db:
            assert db.query(VectorIndexJob).count() == 1


def test_reid_queue_batches_and_isolates_missing_files(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "test-reid-queue-batch",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        REID_INDEX_ON_INGEST="true",
        MILVUS_ENABLED="true",
    )

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.models.vectors import VectorIndexJob, VLEmbedding
    from app.services.reid import ReidEmbeddingService
    from app.services.vector_index import MilvusVectorIndex
    from app.services.vector_index_queue import vector_index_queue

    settings = get_settings()
    # enqueue() auto-starts the worker thread; keep it out of the way so the claims below are
    # deterministic and the batch call count is ours alone.
    monkeypatch.setattr(vector_index_queue, "start", lambda settings=None: None)
    with TestClient(main.create_app()):
        pass  # lifespan creates the tables
    crops_dir = tmp_path / "data" / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    batch_calls: list[int] = []
    upserts: list[str] = []

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)
    monkeypatch.setattr(
        MilvusVectorIndex,
        "upsert_vector",
        lambda self, ot, oid, vec, content="", flush=True: upserts.append(str(oid)),
    )
    monkeypatch.setattr(MilvusVectorIndex, "flush", lambda self, ot: None)

    def fake_embed_images(self, paths):
        batch_calls.append(len(paths))
        return [[0.1] * self.dim for _ in paths]

    monkeypatch.setattr(ReidEmbeddingService, "embed_images", fake_embed_images)

    ids = {}
    with SessionLocal() as db:
        image = Image(image_url="/data/frames/q.jpg", source_type="stream_frame")
        db.add(image)
        db.commit()
        db.refresh(image)
        for name, write_file in (("ok1", True), ("ok2", True), ("gone", False)):
            if write_file:
                (crops_dir / (name + ".jpg")).write_bytes(b"img")
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/" + name + ".jpg",
                bbox={"label": "person"},
            )
            db.add(crop)
            db.commit()
            db.refresh(crop)
            ids[name] = crop.id

    for crop_id in ids.values():
        assert vector_index_queue.enqueue_reid_crop(crop_id) is True

    claimed = vector_index_queue._claim_jobs(settings)
    assert len(claimed) == 3
    vector_index_queue._run_reid_jobs(claimed, settings)

    # One /embed-batch call covering both readable crops, not one HTTP call per crop.
    assert batch_calls == [2]
    assert sorted(upserts) == sorted(str(ids[n]) for n in ("ok1", "ok2"))

    with SessionLocal() as db:
        remaining = {str(job.object_id): job for job in db.query(VectorIndexJob).all()}
        markers = list(
            db.query(VLEmbedding).filter(VLEmbedding.object_type == "reid_person_crop")
        )
    # Indexed jobs are gone; the missing-file job stays for retry with the reason recorded.
    assert set(remaining) == {str(ids["gone"])}
    assert remaining[str(ids["gone"])].attempts == 1
    assert "missing" in remaining[str(ids["gone"])].last_error
    assert {str(m.object_id) for m in markers} == {str(ids["ok1"]), str(ids["ok2"])}
    # Markers carry the full space fingerprint, not just the model name.
    assert all(m.embedding_model.startswith(settings.reid_model + "|") for m in markers)


def test_model_switch_requeues_and_replaces_markers(monkeypatch, tmp_path):
    """Markers from another checkpoint must not count as coverage for the current one."""

    main = load_app(
        monkeypatch,
        tmp_path,
        "test-reid-model-switch",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.models.vectors import VLEmbedding
    from app.services.reid import ReidEmbeddingService
    from app.services.reid_index import ReidIndexService
    from app.services.vector_index import MilvusVectorIndex

    with TestClient(main.create_app()):
        pass  # lifespan creates the tables
    crops_dir = tmp_path / "data" / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    (crops_dir / "c.jpg").write_bytes(b"img")

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)
    monkeypatch.setattr(
        MilvusVectorIndex,
        "upsert_vector",
        lambda self, ot, oid, vec, content="", flush=True: None,
    )
    monkeypatch.setattr(MilvusVectorIndex, "flush", lambda self, ot: None)
    monkeypatch.setattr(
        ReidEmbeddingService,
        "embed_images",
        lambda self, paths: [[0.1] * self.dim for _ in paths],
    )

    base = get_settings()
    with SessionLocal() as db:
        image = Image(image_url="/data/frames/m.jpg", source_type="stream_frame")
        db.add(image)
        db.commit()
        db.refresh(image)
        crop = PersonCrop(image_id=image.id, crop_url="/data/crops/c.jpg", bbox={})
        db.add(crop)
        db.commit()
        db.refresh(crop)

        old = ReidIndexService(db, base)
        assert old.pending_count() == 1
        old.index_crops_batch([crop])
        db.commit()
        assert old.pending_count() == 0

        switched = base.model_copy(update={"reid_model": "sapiensid_wb4m"})
        new = ReidIndexService(db, switched)
        # The wb12m marker does not cover wb4m.
        assert new.pending_count() == 1
        new.index_crops_batch([crop])
        db.commit()
        assert new.pending_count() == 0

        markers = list(
            db.query(VLEmbedding).filter(VLEmbedding.object_type == "reid_person_crop")
        )
    # The stale marker was replaced, not accumulated.
    assert len(markers) == 1
    assert markers[0].embedding_model.startswith("sapiensid_wb4m|")


def test_backfill_does_not_mark_partial_failed_batch_as_indexed(monkeypatch, tmp_path):
    """A partial second Milvus batch must remain wholly pending in durable SQL coverage."""

    main = load_app(
        monkeypatch,
        tmp_path,
        "test-reid-backfill-partial-batch",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
        EMBEDDING_PROVIDER="none",
        VISUAL_EMBEDDING_PROVIDER="none",
    )

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.models.vectors import VLEmbedding
    from app.services.reid import ReidEmbeddingService
    from app.services.reid_index import REID_OBJECT_TYPE, ReidIndexService
    from app.services.vector_index import MilvusVectorIndex, VectorIndexError

    with TestClient(main.create_app()):
        pass

    crops_dir = tmp_path / "data" / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        ReidEmbeddingService,
        "embed_images",
        lambda self, paths: [[0.1] * self.dim for _ in paths],
    )

    upsert_attempts = []

    def fail_on_second_crop_of_second_batch(
        _self,
        _object_type,
        object_id,
        _vector,
        _content="",
        *,
        flush=True,
    ):
        del flush
        upsert_attempts.append(object_id)
        if len(upsert_attempts) == 18:
            raise VectorIndexError("Milvus failed on crop 18")

    monkeypatch.setattr(
        MilvusVectorIndex,
        "upsert_vector",
        fail_on_second_crop_of_second_batch,
    )
    monkeypatch.setattr(MilvusVectorIndex, "flush", lambda self, ot: None)

    settings = get_settings()
    with SessionLocal() as db:
        image = Image(image_url="/data/frames/backfill.jpg", source_type="stream_frame")
        db.add(image)
        db.flush()
        for index in range(18):
            filename = f"backfill-{index:02d}.jpg"
            (crops_dir / filename).write_bytes(b"img")
            db.add(
                PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/{filename}",
                    bbox={"label": "person"},
                )
            )
        db.commit()

        service = ReidIndexService(db, settings)
        result = service.backfill(18)
        markers = list(
            db.query(VLEmbedding).filter(
                VLEmbedding.object_type == REID_OBJECT_TYPE,
                VLEmbedding.embedding_model == service.fingerprint,
                VLEmbedding.embedding_dim == settings.reid_embedding_dim,
            )
        )

    assert len(upsert_attempts) == 18
    assert result["seen"] == 18
    assert result["indexed"] == 16
    assert result["unprocessed"] == 2
    assert len(markers) == 16


def test_concurrent_backfills_keep_one_marker_and_observation(monkeypatch, tmp_path):
    """Two rebuild callers may repeat external upserts but SQL coverage stays idempotent."""

    main = load_app(
        monkeypatch,
        tmp_path,
        "test-reid-concurrent-backfill",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop, PersonObservationIndex
    from app.models.vectors import VLEmbedding
    from app.services.reid import ReidEmbeddingService
    from app.services.reid_index import REID_OBJECT_TYPE, ReidIndexService
    from app.services.vector_index import MilvusVectorIndex
    from app.services.vector_index_queue import vector_index_queue

    monkeypatch.setattr(vector_index_queue, "start", lambda settings=None: None)
    with TestClient(main.create_app()):
        pass

    crops_dir = tmp_path / "data" / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    (crops_dir / "concurrent.jpg").write_bytes(b"img")
    with SessionLocal() as db:
        image = Image(image_url="/data/frames/concurrent.jpg", source_type="stream_frame")
        db.add(image)
        db.flush()
        crop = PersonCrop(
            image_id=image.id,
            crop_url="/data/crops/concurrent.jpg",
            bbox={"label": "person"},
        )
        db.add(crop)
        db.commit()
        crop_id = crop.id

    monkeypatch.setattr(
        ReidEmbeddingService,
        "embed_images",
        lambda self, paths: [[0.1] * self.dim for _ in paths],
    )
    monkeypatch.setattr(
        MilvusVectorIndex,
        "upsert_vector",
        lambda self, ot, oid, vec, content="", flush=True: None,
    )
    external_barrier = threading.Barrier(2)

    def synchronized_flush(self, object_type):
        external_barrier.wait(timeout=3)

    monkeypatch.setattr(MilvusVectorIndex, "flush", synchronized_flush)

    settings = get_settings()
    outcomes: list[dict[str, object]] = []
    errors: list[Exception] = []

    def rebuild() -> None:
        try:
            with SessionLocal() as db:
                outcomes.append(ReidIndexService(db, settings).backfill(1))
        except Exception as exc:  # preserve thread failures for the parent assertion
            errors.append(exc)

    workers = [threading.Thread(target=rebuild) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)

    assert all(worker.is_alive() is False for worker in workers)
    assert errors == []
    assert len(outcomes) == 2
    assert all(result["indexed"] == 1 for result in outcomes)
    with SessionLocal() as db:
        marker_count = (
            db.query(VLEmbedding)
            .filter(
                VLEmbedding.object_type == REID_OBJECT_TYPE,
                VLEmbedding.object_id == crop_id,
            )
            .count()
        )
        observation_count = (
            db.query(PersonObservationIndex)
            .filter(PersonObservationIndex.crop_id == crop_id)
            .count()
        )
    assert marker_count == 1
    assert observation_count == 1
