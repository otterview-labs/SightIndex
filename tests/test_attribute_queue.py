"""VLM inference is independent of capture/indexing and has durable, fenced retries."""

import pytest
from sqlalchemy import update
from test_vector_index_queue_durability import _create_crop, _enqueue_and_claim, load_app


@pytest.fixture
def wired(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "attribute-jobs",
        VLM_STRUCTURED_BACKGROUND="true",
        VLM_PROVIDER="openai_compatible",
        MILVUS_ENABLED="false",
    )
    from app.db.session import SessionLocal
    from app.services.vector_index_queue import ATTRIBUTE_TARGET, VectorIndexQueue

    main.init_db()
    settings = main.get_settings()
    queue = VectorIndexQueue(targets=(ATTRIBUTE_TARGET,))
    crop_id = _create_crop(SessionLocal)
    yield settings, queue, SessionLocal, crop_id


def test_attribute_worker_success_is_atomic_and_not_claimed_by_index_worker(wired, monkeypatch):
    from app.models.media import PersonCrop
    from app.models.vectors import VectorIndexJob
    from app.services.structured_attributes import StructuredAttributeService
    from app.services.vector_index_queue import ATTRIBUTE_TARGET, VectorIndexQueue

    settings, queue, SessionLocal, crop_id = wired
    with SessionLocal() as db:
        queue.enqueue_in_session(db, ATTRIBUTE_TARGET, crop_id, settings)
        db.commit()
    assert VectorIndexQueue()._claim_jobs(settings) == []
    job = queue._claim_jobs(settings)[0]

    def analyze(self, crop, *, persist):
        assert not persist
        assert not self.db.in_transaction(), "VLM must not hold a database transaction open"
        return {"source": "vlm", "clothing": {"upper_color": "blue"}}

    monkeypatch.setattr(StructuredAttributeService, "analyze_person_crop", analyze)
    queue._run_attribute_jobs([job], settings)
    with SessionLocal() as db:
        assert db.query(VectorIndexJob).count() == 0
        assert db.get(PersonCrop, crop_id).attributes["source"] == "vlm"


def test_attribute_failure_survives_restart_and_reconciliation_does_not_reset_it(
    wired, monkeypatch
):
    from app.models.media import PersonCrop
    from app.models.vectors import VectorIndexJob
    from app.services.structured_attributes import StructuredAttributeService
    from app.services.vector_index_queue import ATTRIBUTE_TARGET, VectorIndexQueue

    settings, queue, SessionLocal, crop_id = wired
    job = _enqueue_and_claim(queue, SessionLocal, settings, ATTRIBUTE_TARGET, crop_id)

    def unavailable(*args, **kwargs):
        raise RuntimeError("VLM temporarily unavailable")

    monkeypatch.setattr(StructuredAttributeService, "analyze_person_crop", unavailable)
    queue._run_attribute_jobs([job], settings)
    restarted = VectorIndexQueue(targets=(ATTRIBUTE_TARGET,))
    restarted._reconcile_attributes(settings)
    with SessionLocal() as db:
        remaining = db.query(VectorIndexJob).one()
        assert remaining.status == "pending"
        assert remaining.attempts == 1
        assert remaining.next_run_at is not None
        assert "temporarily unavailable" in remaining.last_error
        assert not db.get(PersonCrop, crop_id).attributes
        remaining.status = "failed"
        db.commit()
    restarted._reconcile_attributes(settings)
    with SessionLocal() as db:
        assert db.query(VectorIndexJob).one().status == "failed"


def test_attribute_worker_cannot_commit_after_lease_is_stolen(wired, monkeypatch):
    from app.models.media import PersonCrop
    from app.models.vectors import VectorIndexJob
    from app.services.structured_attributes import StructuredAttributeService
    from app.services.vector_index_queue import ATTRIBUTE_TARGET

    settings, queue, SessionLocal, crop_id = wired
    job = _enqueue_and_claim(queue, SessionLocal, settings, ATTRIBUTE_TARGET, crop_id)

    def stolen(self, crop, *, persist):
        with SessionLocal() as db:
            db.execute(
                update(VectorIndexJob)
                .where(VectorIndexJob.id == job.id)
                .values(lease_owner="new-worker")
            )
            db.commit()
        return {"source": "vlm"}

    monkeypatch.setattr(StructuredAttributeService, "analyze_person_crop", stolen)
    queue._run_attribute_jobs([job], settings)
    with SessionLocal() as db:
        assert not db.get(PersonCrop, crop_id).attributes
        assert db.query(VectorIndexJob).one().lease_owner == "new-worker"


def test_reconciliation_discovers_historical_untagged_crops(wired):
    from app.models.vectors import VectorIndexJob

    settings, queue, SessionLocal, crop_id = wired
    queue._reconcile_attributes(settings)
    queue._reconcile_attributes(settings)
    with SessionLocal() as db:
        assert db.query(VectorIndexJob).one().object_id == crop_id


def test_full_reconciliation_still_allows_existing_queue_to_drain(wired):
    from app.services.vector_index_queue import ATTRIBUTE_TARGET

    settings, queue, SessionLocal, crop_id = wired
    settings = settings.model_copy(update={"vector_index_background_max_queue": 1})
    with SessionLocal() as db:
        queue.enqueue_in_session(db, ATTRIBUTE_TARGET, crop_id, settings)
        db.commit()
    _create_crop(SessionLocal, url="/data/crops/second.jpg")
    queue._reconcile_attributes(settings)
    assert queue._claim_jobs(settings)[0].object_id == crop_id


def test_failed_job_can_be_explicitly_retried_but_completed_crop_is_not_overwritten(
    wired, monkeypatch
):
    from app.api.attributes import attribute_jobs, retry_attribute_job
    from app.models.media import PersonCrop
    from app.models.vectors import VectorIndexJob
    from app.services.vector_index_queue import ATTRIBUTE_TARGET, attribute_queue

    settings, queue, SessionLocal, crop_id = wired
    monkeypatch.setattr(attribute_queue, "wake", lambda *args: None)
    with SessionLocal() as db:
        queue.enqueue_in_session(db, ATTRIBUTE_TARGET, crop_id, settings)
        db.flush()
        job = db.query(VectorIndexJob).one()
        job.status, job.attempts = "failed", 4
        db.commit()
        assert attribute_jobs(db)["queue"] == {"failed": 1}
        assert retry_attribute_job(crop_id, db, settings)["status"] == "queued"
        db.refresh(job)
        assert job.status == "pending" and job.attempts == 0
        crop = db.get(PersonCrop, crop_id)
        crop.attributes = {"source": "vlm"}
        db.commit()
        assert retry_attribute_job(crop_id, db, settings)["status"] == "already_completed"


def test_full_attribute_queue_does_not_discard_capture(wired, monkeypatch):
    from app.models.media import Image, PersonCrop
    from app.services.frame_processing import FrameProcessingService
    from app.services.vector_index_queue import (
        ATTRIBUTE_TARGET,
        VectorQueueFullError,
        vector_index_queue,
    )

    settings, queue, SessionLocal, crop_id = wired

    def enqueue(db, requests, settings):
        if any(target == ATTRIBUTE_TARGET for target, _ in requests):
            raise VectorQueueFullError("label backlog full")
        return False

    monkeypatch.setattr(vector_index_queue, "enqueue_many_in_session", enqueue)
    with SessionLocal() as db:
        crop = db.get(PersonCrop, crop_id)
        image = db.get(Image, crop.image_id)
        assert not FrameProcessingService(db, settings)._enqueue_index_jobs(image, [crop])
        db.commit()
        assert db.get(PersonCrop, crop_id) is not None
