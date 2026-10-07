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
        assert remaining.last_error == "attribute_error:unknown"
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
        assert attribute_jobs(db, settings)["queue"] == {"pending": 0, "running": 0, "failed": 1}
        assert retry_attribute_job(crop_id, db, settings)["status"] == "queued"
        db.refresh(job)
        assert job.status == "pending" and job.attempts == 4
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


def _failed_job(queue, sessions, settings, crop_id, *, attempts=4, error="attribute_error:timeout"):
    from app.models.vectors import VectorIndexJob
    from app.services.vector_index_queue import ATTRIBUTE_TARGET

    with sessions() as db:
        queue.enqueue_in_session(db, ATTRIBUTE_TARGET, crop_id, settings)
        db.flush()
        row = db.query(VectorIndexJob).one()
        row.status, row.attempts, row.last_error = "failed", attempts, error
        db.commit()
        return row.id


def test_manual_retry_preserves_failure_evidence_and_is_idempotent(wired, monkeypatch):
    from app.api.attributes import retry_attribute_job
    from app.models.vectors import VectorIndexJob
    from app.services.vector_index_queue import attribute_queue

    settings, queue, sessions, crop_id = wired
    job_id = _failed_job(queue, sessions, settings, crop_id)
    calls = []
    monkeypatch.setattr(attribute_queue, "wake", lambda *args: calls.append(True))
    with sessions() as db:
        assert retry_attribute_job(crop_id, db, settings) == {"status": "queued"}
        assert retry_attribute_job(crop_id, db, settings) == {"status": "already_queued"}
        row = db.get(VectorIndexJob, job_id)
        db.refresh(row)
        assert row.attempts == 4 and row.last_error == "attribute_error:timeout"
        assert row.lease_owner is None and row.lease_expires_at is None
        row.status, row.lease_owner = "running", "active-worker"
        db.commit()
        assert retry_attribute_job(crop_id, db, settings) == {"status": "in_progress"}
        db.refresh(row)
        assert row.lease_owner == "active-worker" and row.attempts == 4
    assert calls == [True]


def test_ordinary_attribute_enqueue_does_not_reactivate_terminal_failure(wired):
    from app.models.vectors import VectorIndexJob
    from app.services.vector_index_queue import ATTRIBUTE_TARGET

    settings, queue, sessions, crop_id = wired
    job_id = _failed_job(queue, sessions, settings, crop_id)
    with sessions() as db:
        assert queue.enqueue_in_session(db, ATTRIBUTE_TARGET, crop_id, settings) is False
        db.commit()
        row = db.get(VectorIndexJob, job_id)
        assert row.status == "failed" and row.attempts == 4
        assert row.last_error == "attribute_error:timeout"


def test_explicit_retry_failure_returns_directly_to_failed_without_new_retry_round(
    wired, monkeypatch, caplog
):
    from app.api.attributes import retry_attribute_job
    from app.models.vectors import VectorIndexJob
    from app.services.structured_attributes import StructuredAttributeService
    from app.services.vector_index_queue import attribute_queue

    settings, queue, sessions, crop_id = wired
    job_id = _failed_job(queue, sessions, settings, crop_id)
    monkeypatch.setattr(attribute_queue, "wake", lambda *args: None)
    with sessions() as db:
        assert retry_attribute_job(crop_id, db, settings)["status"] == "queued"
    job = queue._claim_jobs(settings)[0]

    def unavailable(*args, **kwargs):
        raise TimeoutError("secret-token https://private.test/?token=secret /private/file.jpg")

    monkeypatch.setattr(StructuredAttributeService, "analyze_person_crop", unavailable)
    queue._run_attribute_jobs([job], settings)
    assert queue._claim_jobs(settings) == []
    with sessions() as db:
        row = db.get(VectorIndexJob, job_id)
        assert row.status == "failed" and row.attempts == 5
        assert row.last_error == "attribute_error:timeout" and row.next_run_at is None
    assert "secret" not in caplog.text and "/private/" not in caplog.text


@pytest.mark.parametrize("attempts,max_retries", [(1, 3), (4, 4)])
def test_budget_inconsistent_retry_is_rejected_without_erasing_evidence(
    wired, monkeypatch, attempts, max_retries
):
    from fastapi import HTTPException

    from app.api.attributes import retry_attribute_job
    from app.models.vectors import VectorIndexJob
    from app.services.vector_index_queue import attribute_queue

    settings, queue, sessions, crop_id = wired
    job_id = _failed_job(queue, sessions, settings, crop_id, attempts=attempts)
    settings = settings.model_copy(update={"vector_index_background_max_retries": max_retries})
    monkeypatch.setattr(attribute_queue, "wake", lambda *args: pytest.fail("must not wake"))
    with sessions() as db:
        with pytest.raises(HTTPException) as error:
            retry_attribute_job(crop_id, db, settings)
        assert error.value.status_code == 409
        row = db.get(VectorIndexJob, job_id)
        assert (row.status, row.attempts, row.last_error) == (
            "failed",
            attempts,
            "attribute_error:timeout",
        )


def test_queue_full_retry_rolls_back_and_does_not_wake(wired, monkeypatch):
    from fastapi import HTTPException

    from app.api.attributes import retry_attribute_job
    from app.models.vectors import VectorIndexJob
    from app.services.vector_index_queue import ATTRIBUTE_TARGET, attribute_queue

    settings, queue, sessions, crop_id = wired
    job_id = _failed_job(queue, sessions, settings, crop_id)
    second = _create_crop(sessions, url="/data/crops/second.jpg")
    settings = settings.model_copy(update={"vector_index_background_max_queue": 1})
    with sessions() as db:
        queue.enqueue_in_session(db, ATTRIBUTE_TARGET, second, settings)
        db.commit()
    monkeypatch.setattr(attribute_queue, "wake", lambda *args: pytest.fail("must not wake"))
    with sessions() as db:
        with pytest.raises(HTTPException) as error:
            retry_attribute_job(crop_id, db, settings)
        assert error.value.status_code == 503
        row = db.get(VectorIndexJob, job_id)
        assert (row.status, row.attempts, row.last_error) == (
            "failed",
            4,
            "attribute_error:timeout",
        )


def test_missing_job_and_disabled_worker_do_not_create_or_retry_jobs(wired, monkeypatch):
    from fastapi import HTTPException

    from app.api.attributes import retry_attribute_job
    from app.models.vectors import VectorIndexJob
    from app.services.vector_index_queue import attribute_queue

    settings, queue, sessions, crop_id = wired
    monkeypatch.setattr(attribute_queue, "wake", lambda *args: pytest.fail("must not wake"))
    with sessions() as db:
        with pytest.raises(HTTPException) as error:
            retry_attribute_job(crop_id, db, settings)
        assert error.value.status_code == 409
        assert db.query(VectorIndexJob).count() == 0
    job_id = _failed_job(queue, sessions, settings, crop_id)
    with sessions() as db:
        with pytest.raises(HTTPException) as error:
            retry_attribute_job(
                crop_id, db, settings.model_copy(update={"vlm_structured_background": False})
            )
        assert error.value.status_code == 409
        assert db.get(VectorIndexJob, job_id).status == "failed"


def test_jobs_list_masks_old_error_without_rewriting_record(wired):
    from app.api.attributes import attribute_jobs
    from app.models.vectors import VectorIndexJob

    settings, queue, sessions, crop_id = wired
    raw = "secret-key https://private.test/?token=secret /private/known.jpg"
    job_id = _failed_job(queue, sessions, settings, crop_id, error=raw)
    with sessions() as db:
        response = attribute_jobs(db, settings)
        failure = response["failures"][0]
        assert failure["error_code"] == "unknown" and failure["attempts"] == 4
        assert failure["retry_budget_consistent"] is True
        assert failure["updated_at"] is not None
        assert response["worker_configured"] is True
        assert "secret" not in repr(response) and "/private/" not in repr(response)
        assert db.get(VectorIndexJob, job_id).last_error == raw


def test_concurrent_manual_retry_grants_only_one_attempt(wired):
    from concurrent.futures import ThreadPoolExecutor

    from app.models.vectors import VectorIndexJob

    settings, queue, sessions, crop_id = wired
    job_id = _failed_job(queue, sessions, settings, crop_id)

    def retry():
        with sessions() as db:
            result = queue.retry_failed_attribute_in_session(db, crop_id, settings)
            db.commit()
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: retry(), range(2))) == ["already_queued", "queued"]
    with sessions() as db:
        row = db.get(VectorIndexJob, job_id)
        assert row.status == "pending" and row.attempts == 4
        assert row.last_error == "attribute_error:timeout"


def test_manual_retry_success_updates_only_requested_crop_and_acknowledges_job(wired, monkeypatch):
    from app.api.attributes import retry_attribute_job
    from app.models.media import PersonCrop
    from app.models.vectors import VectorIndexJob
    from app.services.structured_attributes import StructuredAttributeService
    from app.services.vector_index_queue import attribute_queue

    settings, queue, sessions, crop_id = wired
    _failed_job(queue, sessions, settings, crop_id)
    unrelated_id = _create_crop(sessions, url="/data/crops/unrelated.jpg")
    monkeypatch.setattr(attribute_queue, "wake", lambda *args: None)
    calls = []

    def analyze(self, crop, *, persist):
        assert crop.id == crop_id and persist is False
        calls.append(crop.id)
        return {"source": "vlm", "clothing": {"upper_color": "blue"}}

    monkeypatch.setattr(StructuredAttributeService, "analyze_person_crop", analyze)
    with sessions() as db:
        assert retry_attribute_job(crop_id, db, settings)["status"] == "queued"
    queue._run_attribute_jobs(queue._claim_jobs(settings), settings)
    with sessions() as db:
        assert db.get(PersonCrop, crop_id).attributes["source"] == "vlm"
        assert db.get(PersonCrop, unrelated_id).attributes is None
        assert db.query(VectorIndexJob).count() == 0
    assert calls == [crop_id]


def test_attribute_analyze_api_errors_are_safe_for_wrapped_upstream_exceptions(wired, monkeypatch):
    from io import BytesIO

    from fastapi import HTTPException, UploadFile

    from app.api.attributes import analyze_attributes, analyze_person_crop_attributes
    from app.services.structured_attributes import StructuredAttributeService
    from app.services.vlm import VLMRuntimeError

    settings, queue, sessions, crop_id = wired

    def upstream(*args, **kwargs):
        error = VLMRuntimeError("https://private.test/?token=secret /private/file.jpg")
        error.__cause__ = TimeoutError("private-secret")
        raise error

    monkeypatch.setattr(StructuredAttributeService, "analyze_person_crop", upstream)
    monkeypatch.setattr(StructuredAttributeService, "analyze_file", upstream)
    with sessions() as db:
        with pytest.raises(HTTPException) as crop_error:
            analyze_person_crop_attributes(crop_id, db, settings, persist=False)
        with pytest.raises(HTTPException) as upload_error:
            analyze_attributes(
                db,
                settings,
                UploadFile(file=BytesIO(b"synthetic"), filename="x.jpg"),
                "person",
                None,
            )
    for error in (crop_error.value, upload_error.value):
        assert error.status_code == 400
        assert error.detail == "The attribute service timed out."
        assert "secret" not in error.detail
