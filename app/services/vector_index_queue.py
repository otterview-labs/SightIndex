from __future__ import annotations

import logging
import random
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import TracebackType
from typing import Self

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.db.session import SessionLocal
from app.models.media import Image, PersonCrop
from app.models.vectors import VectorIndexCapacityLock, VectorIndexJob
from app.services.embeddings import EmbeddingRuntimeError
from app.services.reid_index import BATCH_SIZE as REID_BATCH_SIZE
from app.services.reid_index import REID_OBJECT_TYPE, BatchResult
from app.services.time_utils import local_now
from app.services.vector_index import VectorIndexError, VectorIndexingService

logger = logging.getLogger(__name__)

VL_TARGETS = ("image", "person_crop")
ALL_TARGETS = (*VL_TARGETS, REID_OBJECT_TYPE)
ACTIVE_STATUSES = ("pending", "running")


class VectorQueueFullError(RuntimeError):
    """Raised when a complete atomic outbox set cannot fit in the persistent queue."""


class LeaseLostError(RuntimeError):
    """Raised when this worker can no longer prove ownership of a claimed job."""


@dataclass(frozen=True)
class ClaimedVectorIndexJob:
    """Immutable claim token carried through external side effects and SQL acknowledgement."""

    id: uuid.UUID
    target: str
    object_id: uuid.UUID
    attempts: int
    lease_owner: str


@dataclass
class _LeaseHeartbeat:
    """Renew a claim while slow model and Milvus calls execute outside SQL transactions."""

    queue: VectorIndexQueue
    jobs: tuple[ClaimedVectorIndexJob, ...]
    settings: Settings
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _lost_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _fenced: bool = field(default=False, init=False)

    @property
    def owner(self) -> str:
        return self.jobs[0].lease_owner

    @property
    def job_ids(self) -> tuple[uuid.UUID, ...]:
        return tuple(job.id for job in self.jobs)

    def __enter__(self) -> Self:
        if not self.queue._renew_leases(self.job_ids, self.owner, self.settings):
            raise LeaseLostError("vector index lease was lost before processing started")
        self._thread = threading.Thread(
            target=self._run,
            name="sightindex-vector-index-lease-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop()

    def fence(self) -> None:
        """Stop renewal and obtain one final full lease before committing SQL state."""

        if not self._stop():
            raise LeaseLostError("vector index lease heartbeat did not stop before fencing")
        if self._lost_event.is_set() or not self.queue._renew_leases(
            self.job_ids, self.owner, self.settings
        ):
            raise LeaseLostError("vector index lease was lost during external processing")
        self._fenced = True

    def _run(self) -> None:
        interval = max(0.5, min(10.0, self.settings.vector_index_lease_seconds / 3.0))
        while not self._stop_event.wait(interval):
            try:
                renewed = self.queue._renew_leases(self.job_ids, self.owner, self.settings)
            except Exception:
                logger.exception("vector index lease heartbeat failed")
                self._lost_event.set()
                return
            if not renewed:
                self._lost_event.set()
                return

    def _stop(self) -> bool:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self.settings.vector_index_lease_seconds))
        if thread is not None and thread.is_alive():
            # Keep the reference and fail fencing. Starting final SQL work while an old,
            # blocked renew call is still alive would make lease ownership ambiguous.
            self._lost_event.set()
            return False
        self._thread = None
        return True


class VectorIndexQueue:
    """Persistent, leased outbox for VL and ReID indexing work."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._settings: Settings | None = None
        self._owner = uuid.uuid4().hex

    def start(self, settings: Settings | None = None) -> None:
        """Start one in-process consumer when at least one complete target is configured."""

        settings = settings or get_settings()
        if not any(self.target_enabled(target, settings) for target in ALL_TARGETS):
            return
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._settings = settings
            self._owner = uuid.uuid4().hex
            self._stop_event.clear()
            try:
                self._recover_expired_leases(settings)
            except Exception:
                logger.exception("vector_index_queue lease recovery failed")
            self._thread = threading.Thread(
                target=self._run,
                name="sightindex-vector-index-queue",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Request shutdown without forgetting a worker that is still alive after timeout."""

        with self._lifecycle_lock:
            self._stop_event.set()
            self._wake_event.set()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lifecycle_lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None
            elif thread is not None and thread.is_alive():
                logger.warning("vector_index_queue worker did not stop within %.1fs", timeout)

    def wake(self, settings: Settings | None = None) -> None:
        """Wake the consumer only after the caller commits its durable outbox rows."""

        self.start(settings)
        self._wake_event.set()

    def enqueue_image(self, image_id: uuid.UUID) -> bool:
        return self.enqueue("image", image_id)

    def enqueue_crop(self, crop_id: uuid.UUID) -> bool:
        return self.enqueue("person_crop", crop_id)

    def enqueue_reid_crop(self, crop_id: uuid.UUID) -> bool:
        return self.enqueue(REID_OBJECT_TYPE, crop_id)

    def target_enabled(self, target: str, settings: Settings) -> bool:
        """Return whether a target has every prerequisite needed for consumption."""

        if target not in ALL_TARGETS:
            return False
        if target == REID_OBJECT_TYPE:
            return bool(
                settings.reid_enabled
                and settings.reid_service_url
                and settings.reid_index_on_ingest
                and settings.milvus_enabled
                and settings.milvus_metric_type.strip().upper() == "COSINE"
            )
        has_vl_provider = (
            settings.embedding_provider != "none"
            or settings.visual_embedding_provider != "none"
        )
        return bool(
            settings.vector_index_on_ingest
            and settings.vector_index_on_ingest_background
            and settings.milvus_enabled
            and has_vl_provider
        )

    def enqueue_many_in_session(
        self,
        db: Session,
        requests: list[tuple[str, uuid.UUID]],
        settings: Settings | None = None,
    ) -> bool:
        """Stage an all-or-nothing set of job rows in the caller's transaction.

        Capacity is enforced per target so a disabled target's backlog cannot starve a newly
        enabled target. No rows are mutated until every requested target has enough capacity.
        """

        settings = settings or self._settings or get_settings()
        unique_requests = list(dict.fromkeys(requests))
        for target, _ in unique_requests:
            if target not in ALL_TARGETS:
                raise ValueError(f"Unsupported vector index target: {target}")
        enabled_requests = [
            request
            for request in unique_requests
            if self.target_enabled(request[0], settings)
        ]
        if not enabled_requests:
            return False

        db.flush()
        self._lock_capacity_targets_in_session(
            db,
            tuple(sorted({target for target, _ in enabled_requests})),
        )

        conditions = [
            and_(VectorIndexJob.target == target, VectorIndexJob.object_id == object_id)
            for target, object_id in enabled_requests
        ]
        existing_rows = list(db.scalars(select(VectorIndexJob).where(or_(*conditions))))
        existing = {(row.target, row.object_id): row for row in existing_rows}

        additions_by_target: dict[str, int] = {}
        for request in enabled_requests:
            row = existing.get(request)
            if row is None or row.status == "failed":
                additions_by_target[request[0]] = additions_by_target.get(request[0], 0) + 1

        for target, additions in additions_by_target.items():
            queued_count = db.scalar(
                select(func.count())
                .select_from(VectorIndexJob)
                .where(
                    VectorIndexJob.target == target,
                    VectorIndexJob.status.in_(ACTIVE_STATUSES),
                )
            )
            total = int(queued_count or 0) + additions
            if total > settings.vector_index_background_max_queue:
                raise VectorQueueFullError(
                    f"vector index queue target={target} needs {total} active jobs; "
                    f"limit={settings.vector_index_background_max_queue}"
                )

        now = local_now(settings)
        for target, object_id in enabled_requests:
            row = existing.get((target, object_id))
            if row is not None:
                if row.status == "failed":
                    row.status = "pending"
                    row.attempts = 0
                    row.next_run_at = now
                    row.last_error = None
                    row.lease_owner = None
                    row.lease_expires_at = None
                    db.add(row)
                continue
            db.add(
                VectorIndexJob(
                    target=target,
                    object_id=object_id,
                    status="pending",
                    next_run_at=now,
                )
            )
        return True

    def _lock_capacity_targets_in_session(
        self,
        db: Session,
        targets: tuple[str, ...],
    ) -> None:
        """Serialize count + insert reservations across processes and database dialects.

        PostgreSQL holds row locks until transaction end. SQLite's first UPDATE obtains its
        database write reservation, so a concurrent enqueue waits and recounts after commit.
        Fixed target ordering prevents multi-target deadlocks.
        """

        for target in targets:
            result = db.execute(
                update(VectorIndexCapacityLock)
                .where(VectorIndexCapacityLock.target == target)
                .values(revision=VectorIndexCapacityLock.revision + 1)
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                raise RuntimeError(
                    f"vector index capacity lock is missing for target={target}; run init_db()"
                )

    def enqueue_in_session(
        self,
        db: Session,
        target: str,
        object_id: uuid.UUID,
        settings: Settings | None = None,
    ) -> bool:
        """Stage one job in an existing transaction."""

        return self.enqueue_many_in_session(db, [(target, object_id)], settings)

    def enqueue(self, target: str, object_id: uuid.UUID) -> bool:
        """Persist one standalone job and wake the worker after commit."""

        settings = self._settings or get_settings()
        try:
            with SessionLocal() as db:
                try:
                    added = self.enqueue_in_session(db, target, object_id, settings)
                    db.commit()
                except IntegrityError as exc:
                    db.rollback()
                    winner = db.scalar(
                        select(VectorIndexJob.id).where(
                            VectorIndexJob.target == target,
                            VectorIndexJob.object_id == object_id,
                        )
                    )
                    if winner is None:
                        raise exc
                    added = True
        except VectorQueueFullError as exc:
            logger.error("vector_index_queue: %s", exc)
            return False
        except Exception:
            logger.exception(
                "vector_index_queue enqueue failed target=%s object_id=%s",
                target,
                object_id,
            )
            return False
        if added:
            self.wake(settings)
        return added

    def _recover_expired_leases(self, settings: Settings) -> None:
        now = local_now(settings)
        with SessionLocal() as db:
            db.execute(
                update(VectorIndexJob)
                .where(
                    VectorIndexJob.status == "running",
                    or_(
                        VectorIndexJob.lease_expires_at.is_(None),
                        VectorIndexJob.lease_expires_at <= now,
                    ),
                )
                .values(
                    status="pending",
                    lease_owner=None,
                    lease_expires_at=None,
                    next_run_at=now,
                )
            )
            db.commit()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                settings = self._settings or get_settings()
                jobs = self._claim_jobs(settings)
                if not jobs:
                    self._wake_event.wait(settings.vector_index_background_idle_seconds)
                    self._wake_event.clear()
                    continue
                if jobs[0].target == REID_OBJECT_TYPE:
                    self._run_reid_jobs(jobs, settings)
                else:
                    self._run_vl_jobs(jobs, settings)
            except Exception:
                logger.exception("vector_index_queue worker loop error")
                self._stop_event.wait(1.0)

    def _run_vl_jobs(self, jobs: list[ClaimedVectorIndexJob], settings: Settings) -> None:
        for job in jobs:
            try:
                with _LeaseHeartbeat(self, (job,), settings) as lease:
                    object_exists = self._write_vl_vector(job, settings)
                    lease.fence()
                    self._commit_vl_success(job, settings, object_exists=object_exists)
            except LeaseLostError as exc:
                logger.warning("vector_index_queue lost VL lease job=%s: %s", job.id, exc)
            except (EmbeddingRuntimeError, VectorIndexError) as exc:
                self._mark_failed(job, str(exc), settings)
            except Exception as exc:
                logger.exception("vector_index_queue VL job failed job=%s", job.id)
                self._mark_failed(job, str(exc), settings)

    def _write_vl_vector(self, job: ClaimedVectorIndexJob, settings: Settings) -> bool:
        """Run only external embedding/Milvus I/O; do not write SQL metadata here."""

        with SessionLocal() as db:
            indexer = VectorIndexingService(db, settings)
            if job.target == "image":
                image = db.get(Image, job.object_id)
                if image is None:
                    return False
                indexer.write_image_vector(image, flush=False)
            elif job.target == "person_crop":
                crop = db.get(PersonCrop, job.object_id)
                if crop is None:
                    return False
                indexer.write_crop_vector(crop, flush=False)
            else:
                raise VectorIndexError(f"Unsupported vector index job target: {job.target}")
            indexer.index.flush(job.target)
            return True

    def _commit_vl_success(
        self,
        job: ClaimedVectorIndexJob,
        settings: Settings,
        *,
        object_exists: bool,
    ) -> None:
        """Commit marker/observation and owner-fenced ack in one short SQL transaction."""

        with SessionLocal() as db:
            self._lock_owned_jobs_in_session(db, (job,), settings)
            if object_exists:
                indexer = VectorIndexingService(db, settings)
                if job.target == "image":
                    image = db.get(Image, job.object_id)
                    if image is not None:
                        indexer.record_image_index(image)
                else:
                    crop = db.get(PersonCrop, job.object_id)
                    if crop is not None:
                        indexer.record_crop_index(crop)
            self._delete_owned_job_in_session(db, job)
            db.commit()

    def _run_reid_jobs(self, jobs: list[ClaimedVectorIndexJob], settings: Settings) -> None:
        from app.services.reid import ReidRuntimeError
        from app.services.reid_index import ReidIndexService

        jobs_by_crop: dict[uuid.UUID, list[ClaimedVectorIndexJob]] = {}
        for job in jobs:
            jobs_by_crop.setdefault(job.object_id, []).append(job)

        try:
            with _LeaseHeartbeat(self, tuple(jobs), settings) as lease:
                with SessionLocal() as db:
                    crops = [
                        crop
                        for crop_id in jobs_by_crop
                        if (crop := db.get(PersonCrop, crop_id)) is not None
                    ]
                    service = ReidIndexService(db, settings)
                    result = service.index_crops_batch(crops, flush=True, record=False)
                lease.fence()
                self._commit_reid_result(jobs_by_crop, result, settings)
        except LeaseLostError as exc:
            logger.warning("vector_index_queue lost ReID lease: %s", exc)
        except (ReidRuntimeError, VectorIndexError) as exc:
            self._mark_failed_many(jobs, str(exc), settings)
        except Exception as exc:
            logger.exception("vector_index_queue ReID batch failed")
            self._mark_failed_many(jobs, str(exc), settings)

    def _commit_reid_result(
        self,
        jobs_by_crop: dict[uuid.UUID, list[ClaimedVectorIndexJob]],
        result: BatchResult,
        settings: Settings,
    ) -> None:
        from app.services.reid_index import ReidIndexService

        indexed = set(result.indexed)
        skipped = set(result.skipped)
        with SessionLocal() as db:
            all_jobs = tuple(job for crop_jobs in jobs_by_crop.values() for job in crop_jobs)
            self._lock_owned_jobs_in_session(db, all_jobs, settings)
            service = ReidIndexService(db, settings)
            for crop_id, crop_jobs in jobs_by_crop.items():
                crop = db.get(PersonCrop, crop_id)
                if crop_id in indexed and crop is not None:
                    service.record_indexed_crop(crop)
                    for job in crop_jobs:
                        self._delete_owned_job_in_session(db, job)
                    continue
                if crop is None:
                    for job in crop_jobs:
                        self._delete_owned_job_in_session(db, job)
                    continue
                error = (
                    "crop file missing"
                    if crop_id in skipped
                    else result.failures.get(crop_id, "ReID batch returned no outcome")
                )
                for job in crop_jobs:
                    if not self._fail_owned_job_in_session(db, job, error, settings):
                        raise LeaseLostError(
                            f"cannot record ReID failure for job {job.id}; lease token changed"
                        )
            db.commit()

    def _claim_jobs(self, settings: Settings) -> list[ClaimedVectorIndexJob]:
        now = local_now(settings)
        enabled_targets = [
            target for target in ALL_TARGETS if self.target_enabled(target, settings)
        ]
        if not enabled_targets:
            return []
        claimable = self._claimable(now)
        owner = self._owner
        lease_until = now + timedelta(seconds=settings.vector_index_lease_seconds)
        with SessionLocal() as db:
            target = db.scalar(
                select(VectorIndexJob.target)
                .where(VectorIndexJob.target.in_(enabled_targets), claimable)
                .order_by(VectorIndexJob.created_at)
                .limit(1)
            )
            if target is None:
                return []
            limit = (
                min(settings.vector_index_background_batch_size, REID_BATCH_SIZE)
                if target == REID_OBJECT_TYPE
                else 1
            )
            candidates = list(
                db.scalars(
                    select(VectorIndexJob)
                    .where(VectorIndexJob.target == target, claimable)
                    .order_by(VectorIndexJob.created_at)
                    .limit(limit)
                )
            )
            claimed: list[ClaimedVectorIndexJob] = []
            for job in candidates:
                result = db.execute(
                    update(VectorIndexJob)
                    .where(VectorIndexJob.id == job.id, self._claimable(now))
                    .values(
                        status="running",
                        lease_owner=owner,
                        lease_expires_at=lease_until,
                        updated_at=now,
                    )
                    .execution_options(synchronize_session=False)
                )
                if result.rowcount == 1:
                    claimed.append(
                        ClaimedVectorIndexJob(
                            id=job.id,
                            target=job.target,
                            object_id=job.object_id,
                            attempts=job.attempts,
                            lease_owner=owner,
                        )
                    )
            db.commit()
            return claimed

    @staticmethod
    def _claimable(now: datetime):
        return or_(
            and_(
                VectorIndexJob.status == "pending",
                or_(
                    VectorIndexJob.next_run_at.is_(None),
                    VectorIndexJob.next_run_at <= now,
                ),
            ),
            and_(
                VectorIndexJob.status == "running",
                VectorIndexJob.lease_expires_at.is_not(None),
                VectorIndexJob.lease_expires_at <= now,
            ),
        )

    def _renew_leases(
        self,
        job_ids: tuple[uuid.UUID, ...],
        owner: str,
        settings: Settings,
    ) -> bool:
        if not job_ids:
            return True
        now = local_now(settings)
        lease_until = now + timedelta(seconds=settings.vector_index_lease_seconds)
        with SessionLocal() as db:
            result = db.execute(
                update(VectorIndexJob)
                .where(
                    VectorIndexJob.id.in_(job_ids),
                    VectorIndexJob.status == "running",
                    VectorIndexJob.lease_owner == owner,
                )
                .values(lease_expires_at=lease_until, updated_at=now)
                .execution_options(synchronize_session=False)
            )
            db.commit()
            return result.rowcount == len(job_ids)

    def _delete_owned_job_in_session(
        self,
        db: Session,
        job: ClaimedVectorIndexJob,
    ) -> None:
        result = db.execute(
            delete(VectorIndexJob).where(
                VectorIndexJob.id == job.id,
                VectorIndexJob.status == "running",
                VectorIndexJob.lease_owner == job.lease_owner,
            )
        )
        if result.rowcount != 1:
            raise LeaseLostError(f"cannot acknowledge job {job.id}; lease owner changed")

    def _lock_owned_jobs_in_session(
        self,
        db: Session,
        jobs: tuple[ClaimedVectorIndexJob, ...],
        settings: Settings,
    ) -> None:
        """Fence final SQL work by locking every still-owned running job row."""

        if not jobs:
            return
        owner = jobs[0].lease_owner
        if any(job.lease_owner != owner for job in jobs):
            raise LeaseLostError("cannot finalize jobs claimed by different owners")
        now = local_now(settings)
        result = db.execute(
            update(VectorIndexJob)
            .where(
                VectorIndexJob.id.in_(tuple(job.id for job in jobs)),
                VectorIndexJob.status == "running",
                VectorIndexJob.lease_owner == owner,
            )
            .values(
                lease_expires_at=now
                + timedelta(seconds=settings.vector_index_lease_seconds),
                updated_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != len(jobs):
            raise LeaseLostError("cannot finalize vector jobs; one or more leases changed")

    def _fail_owned_job_in_session(
        self,
        db: Session,
        job: ClaimedVectorIndexJob,
        error: str,
        settings: Settings,
    ) -> bool:
        # This must be a single owner-fenced UPDATE.  Selecting an ORM row and
        # mutating it later lets SQLAlchemy flush an UPDATE keyed only by the primary
        # key; under READ COMMITTED a stale worker could then overwrite a fresh
        # claimant that acquired the job between SELECT and flush.
        attempts = job.attempts + 1
        status = (
            "failed"
            if attempts > settings.vector_index_background_max_retries
            else "pending"
        )
        next_run_at = None
        if status == "pending":
            base = settings.vector_index_background_retry_delay_seconds
            exponential = min(base * (2 ** max(0, attempts - 1)), 3600.0)
            jitter = random.uniform(0.0, min(30.0, exponential * 0.2))
            next_run_at = local_now(settings) + timedelta(seconds=exponential + jitter)
        now = local_now(settings)
        result = db.execute(
            update(VectorIndexJob)
            .where(
                VectorIndexJob.id == job.id,
                VectorIndexJob.status == "running",
                VectorIndexJob.lease_owner == job.lease_owner,
                VectorIndexJob.attempts == job.attempts,
            )
            .values(
                attempts=attempts,
                last_error=error[:1000],
                lease_owner=None,
                lease_expires_at=None,
                status=status,
                next_run_at=next_run_at,
                updated_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        return result.rowcount == 1

    def _mark_done(self, job_id: uuid.UUID) -> None:
        """Acknowledge a job for focused compatibility tests.

        Production paths acknowledge inside the same transaction as their marker rows.
        """

        with SessionLocal() as db:
            result = db.execute(
                delete(VectorIndexJob).where(
                    VectorIndexJob.id == job_id,
                    VectorIndexJob.status == "running",
                    VectorIndexJob.lease_owner == self._owner,
                )
            )
            if result.rowcount == 1:
                db.commit()
            else:
                db.rollback()

    def _mark_failed(
        self,
        job: ClaimedVectorIndexJob,
        error: str,
        settings: Settings,
    ) -> None:
        self._mark_failed_many([job], error, settings)

    def _mark_failed_many(
        self,
        jobs: list[ClaimedVectorIndexJob],
        error: str,
        settings: Settings,
    ) -> None:
        with SessionLocal() as db:
            for job in jobs:
                self._fail_owned_job_in_session(db, job, error, settings)
            db.commit()

    def _index_job(
        self,
        db: Session,
        indexer: VectorIndexingService,
        job: ClaimedVectorIndexJob,
    ) -> None:
        """Legacy narrow helper retained for callers that do their own transaction handling."""

        if job.target == "image":
            image = db.get(Image, job.object_id)
            if image is not None:
                indexer.index_image(image, flush=False)
            return
        if job.target == "person_crop":
            crop = db.get(PersonCrop, job.object_id)
            if crop is not None:
                indexer.index_crop(crop, flush=False)
            return
        raise VectorIndexError(f"Unsupported vector index job target: {job.target}")

    def stats(self) -> dict[str, int]:
        with SessionLocal() as db:
            rows = db.execute(
                select(VectorIndexJob.status, func.count()).group_by(VectorIndexJob.status)
            )
            result = {"pending": 0, "running": 0, "failed": 0}
            result.update({str(status): int(count) for status, count in rows})
            return result


vector_index_queue = VectorIndexQueue()
