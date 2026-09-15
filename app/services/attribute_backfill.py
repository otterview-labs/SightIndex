import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Query, Session

from app.config.settings import Settings
from app.models.media import PersonCrop
from app.services.structured_attributes import StructuredAttributeService
from app.services.vlm import VLMRuntimeError

logger = logging.getLogger(__name__)

BackfillResult = Literal["updated", "skipped", "unreadable"]
BackfillProcessor = Callable[[PersonCrop], BackfillResult | None]


@dataclass
class AttributeBackfillProgress:
    attempted: int = 0
    updated: int = 0
    skipped: int = 0
    unreadable: int = 0
    remaining: int = 0
    completed: bool = False
    last_crop_id: str | None = None
    # ``last_crop_id`` predates the keyset walker and is kept in the JSON checkpoint/API.
    # The timestamp is the first half of the stable ``(created_at, id)`` cursor.
    last_created_at: str | None = None
    failures: dict[str, int] = field(default_factory=dict)
    permanent_failures: dict[str, str] = field(default_factory=dict)
    # Bounded errors from the most recent invocation keep the legacy HTTP response useful while
    # the durable checkpoint remains the source of truth for retries and permanent failures.
    last_errors: list[str] = field(default_factory=list)
    updated_at: str | None = None


class DurableAttributeBackfillService:
    """Upgrade legacy crop tags with a durable, restart-safe progress checkpoint."""

    # A page is always bounded.  The SQLAlchemy path filters pending rows in the database, so
    # this is also the maximum number of PersonCrop objects materialised by one query.  The
    # fallback path is only for the tiny in-memory fakes used by older callers/tests.
    _PAGE_SIZE = 256

    def __init__(
        self,
        db: Session,
        settings: Settings,
        *,
        state_path: Path | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.structured = StructuredAttributeService(db, settings)
        self.state_path = state_path or (settings.data_dir / "tasks" / "attribute-backfill.json")

    def run(
        self,
        *,
        batch_size: int = 50,
        max_attempts: int = 3,
        force: bool = False,
        processor: BackfillProcessor | None = None,
        include_described: bool | None = None,
        max_items: int | None = None,
    ) -> AttributeBackfillProgress:
        """Process pending crops with a durable keyset cursor.

        ``processor`` is used by legacy backfill endpoints that need the same durable walker but
        have a different local extractor (for example the cheap CV clothing-tone reader).  It
        returns ``"updated"`` by default, ``"skipped"`` for an intentional no-op and
        ``"unreadable"`` for a missing image.  The original VLM path remains the default.

        ``max_items`` limits one invocation while keeping the checkpoint, which preserves the
        old HTTP endpoint's batch semantics.  A normal worker call omits it and drains the
        backlog until it is complete.
        """
        progress = self.load_progress()
        progress.completed = False
        progress.last_errors = []
        initial_attempted = progress.attempted
        if max_items is not None and max_items < 1:
            return progress
        if include_described is None:
            # The VLM path skips existing descriptions unless force is requested.  Custom
            # processors decide this themselves, so they inspect every row and can report a
            # compatible skipped count to callers.
            include_described = force or processor is not None
        if processor is None:
            processor = self._process_vlm
        # Keep the public ``last_crop_id`` field while using a typed timestamp+UUID cursor
        # internally.  A checkpoint written by the old implementation has no timestamp; in that
        # case we deliberately restart from the beginning and let the source=vlm predicate skip
        # already completed rows.
        cursor_created_at, cursor_crop_id = self._checkpoint_cursor(progress)
        wrapped = False
        counted_remaining = False
        while True:
            candidates, remaining = self._pending(
                batch_size,
                excluded=set(progress.permanent_failures),
                after_created_at=cursor_created_at,
                after_crop_id=cursor_crop_id,
                count_remaining=not counted_remaining,
                include_described=include_described,
            )
            if not counted_remaining:
                # One exact count per invocation gives callers a useful starting value.  Later
                # pages deliberately use a cheap lower-bound hint; repeatedly COUNTing a large
                # table made the old durable worker spend more time scanning than analysing.
                progress.remaining = remaining
                counted_remaining = True
            elif candidates and progress.remaining <= 0:
                # Rows can arrive after the initial count (or be found by the wrap-around pass).
                # Re-seed the visible estimate without claiming an exact live count.
                progress.remaining = remaining
            if not candidates:
                # New rows normally have a later created_at and are picked up by the forward
                # cursor.  One bounded wrap-around also catches rows inserted in the same
                # database timestamp bucket with a lower UUID (and old clocks), without making a
                # live capture stream loop forever.
                if (
                    remaining == 0
                    and (cursor_created_at is not None or cursor_crop_id is not None)
                    and not wrapped
                    and max_items is None
                ):
                    cursor_created_at = None
                    cursor_crop_id = None
                    wrapped = True
                    self._save(progress)
                    continue
                progress.completed = remaining == 0
                self._save(progress)
                return progress
            for crop in candidates:
                if max_items is not None and progress.attempted - initial_attempted >= max_items:
                    self._save(progress)
                    return progress
                crop_id = str(crop.id)
                progress.attempted += 1
                try:
                    result = processor(crop)
                except (VLMRuntimeError, OSError, ValueError) as exc:
                    attempts = progress.failures.get(crop_id, 0) + 1
                    progress.failures[crop_id] = attempts
                    if attempts >= max_attempts:
                        progress.permanent_failures[crop_id] = str(exc)
                        progress.remaining = max(0, progress.remaining - 1)
                        cursor_created_at, cursor_crop_id = self._advance_cursor(
                            progress,
                            crop,
                        )
                    logger.warning(
                        "Attribute backfill failed for %s (attempt %s/%s): %s",
                        crop_id,
                        attempts,
                        max_attempts,
                        exc,
                    )
                    progress.last_errors.append(f"{crop_id}: {exc}")
                    del progress.last_errors[:-20]
                    self._save(progress)
                    # Do not move past a retryable item.  This keeps a transiently unavailable
                    # crop in the next keyset page instead of silently losing it after restart.
                    if attempts < max_attempts:
                        break
                else:
                    result = result or "updated"
                    if result == "skipped":
                        progress.skipped += 1
                    elif result == "unreadable":
                        progress.unreadable += 1
                    else:
                        progress.updated += 1
                    progress.failures.pop(crop_id, None)
                    progress.remaining = max(0, progress.remaining - 1)
                    if hasattr(self.db, "commit"):
                        self.db.commit()
                    cursor_created_at, cursor_crop_id = self._advance_cursor(
                        progress,
                        crop,
                    )
                    self._save(progress)

    def load_progress(self) -> AttributeBackfillProgress:
        if not self.state_path.is_file():
            return AttributeBackfillProgress()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("attribute backfill checkpoint must be an object")
            # Ignore fields added by a newer worker so an older binary can still resume the
            # checkpoint.  Missing fields (including last_created_at in legacy checkpoints) use
            # the dataclass defaults.
            allowed_fields = {item.name for item in fields(AttributeBackfillProgress)}
            compatible = {key: value for key, value in payload.items() if key in allowed_fields}
            return AttributeBackfillProgress(**compatible)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable attribute backfill state", exc_info=True)
            return AttributeBackfillProgress()

    def _process_vlm(self, crop: PersonCrop) -> BackfillResult:
        self.structured.analyze_person_crop(crop, persist=True)
        return "updated"

    def _pending(
        self,
        limit: int,
        *,
        excluded: set[str],
        after_created_at: datetime | None = None,
        after_crop_id: uuid.UUID | None = None,
        count_remaining: bool = True,
        include_described: bool = False,
    ) -> tuple[list[PersonCrop], int]:
        if limit < 1:
            return [], 0

        query = self.db.query(PersonCrop)
        if isinstance(query, Query):
            # Keep the source predicate in SQL.  Besides avoiding an unbounded .all(), this
            # makes ``remaining`` an exact count even when most of the table is already VLM
            # enriched.
            predicates = []
            if not include_described:
                predicates.append(
                    func.coalesce(PersonCrop.attributes["source"].as_string(), "") != "vlm"
                )
            if after_created_at is not None:
                # SQLite stores server-default DateTime values at second precision, while the
                # SQLAlchemy bind for a Python datetime includes ``.000000``.  Comparing the
                # textual values directly would make every row in that second disappear after a
                # checkpoint.  julianday gives SQLite a precision-stable numeric key; other
                # databases retain their native timestamp comparison.
                if self.db.get_bind().dialect.name == "sqlite":
                    created_key = func.julianday(PersonCrop.created_at)
                    cursor_key = func.julianday(after_created_at)
                else:
                    created_key = PersonCrop.created_at
                    cursor_key = after_created_at
                if after_crop_id is not None:
                    predicates.append(
                        or_(
                            created_key > cursor_key,
                            and_(
                                created_key == cursor_key,
                                PersonCrop.id > after_crop_id,
                            ),
                        )
                    )
                else:
                    predicates.append(created_key > cursor_key)
            excluded_ids = self._uuid_values(excluded)
            if excluded_ids:
                predicates.append(~PersonCrop.id.in_(excluded_ids))
            filtered = query.filter(*predicates)
            rows = (
                filtered.order_by(
                    PersonCrop.created_at.asc(),
                    PersonCrop.id.asc(),
                )
                .limit(min(limit, self._PAGE_SIZE))
                .all()
            )
            # A concurrent update can turn a row into source=vlm between count and page reads;
            # retain the Python guard as a harmless last line of defence.
            pending = [
                crop
                for crop in rows
                if self._is_after(crop, after_created_at, after_crop_id)
                and self._is_pending(crop, excluded, include_described=include_described)
            ]
            if count_remaining:
                remaining = int(filtered.order_by(None).count())
            else:
                # A bounded lower-bound hint is enough after the one initial count.  The run
                # loop decrements it as work succeeds and sets it to zero only after an empty
                # page, so status remains useful without a repeated table-wide COUNT.
                remaining = len(pending)
            return pending[:limit], max(0, remaining)

        # Compatibility path for the lightweight query doubles used by the original script and
        # tests.  It still requests a bounded page when the double supports ``limit`` and applies
        # the same keyset/pending predicates in Python.
        page_size = min(limit, self._PAGE_SIZE)
        pending: list[PersonCrop] = []
        offset = 0
        while True:
            page_query = self.db.query(PersonCrop)
            try:
                page_query = page_query.order_by(
                    PersonCrop.created_at.asc(),
                    PersonCrop.id.asc(),
                )
            except (AttributeError, TypeError):
                page_query = page_query.order_by(PersonCrop.created_at)
            try:
                page_query = page_query.offset(offset)
            except (AttributeError, TypeError):
                # The original minimal test double has no offset; its one bounded page remains
                # compatible, while real SQLAlchemy sessions take the keyset path above.
                offset_supported = False
            else:
                offset_supported = True
            try:
                page_query = page_query.limit(page_size)
            except (AttributeError, TypeError):
                page_query = page_query
            rows = list(page_query.all())
            page_pending = [
                crop
                for crop in rows
                if self._is_after(crop, after_created_at, after_crop_id)
                and self._is_pending(crop, excluded, include_described=include_described)
            ]
            pending.extend(page_pending)
            if (
                (not count_remaining and len(pending) >= limit)
                or len(rows) < page_size
                or not offset_supported
            ):
                break
            offset += len(rows)
        # In-memory fakes return all rows from bounded pages; preserve the old exact remaining
        # count without ever requiring an unbounded .all().
        return pending[:limit], len(pending)

    @staticmethod
    def _uuid_values(values: set[str]) -> list[uuid.UUID]:
        result: list[uuid.UUID] = []
        for value in values:
            try:
                result.append(uuid.UUID(str(value)))
            except (ValueError, TypeError, AttributeError):
                continue
        return result

    @staticmethod
    def _is_pending(
        crop: PersonCrop,
        excluded: set[str],
        *,
        include_described: bool = False,
    ) -> bool:
        if str(crop.id) in excluded:
            return False
        return include_described or (
            not isinstance(crop.attributes, dict) or crop.attributes.get("source") != "vlm"
        )

    @classmethod
    def _is_after(
        cls,
        crop: PersonCrop,
        after_created_at: datetime | None,
        after_crop_id: uuid.UUID | None,
    ) -> bool:
        if after_created_at is None:
            return True
        created_at = getattr(crop, "created_at", None)
        if created_at is None:
            return False
        left = cls._normalise_datetime(created_at)
        right = cls._normalise_datetime(after_created_at)
        if left > right:
            return True
        if left < right or after_crop_id is None:
            return False
        try:
            return uuid.UUID(str(crop.id)) > after_crop_id
        except (ValueError, TypeError, AttributeError):
            return False

    @staticmethod
    def _normalise_datetime(value: datetime) -> datetime:
        if value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value

    @classmethod
    def _checkpoint_cursor(
        cls,
        progress: AttributeBackfillProgress,
    ) -> tuple[datetime | None, uuid.UUID | None]:
        if not progress.last_created_at:
            return None, None
        try:
            created_at = datetime.fromisoformat(progress.last_created_at)
        except (TypeError, ValueError):
            return None, None
        try:
            crop_id = uuid.UUID(progress.last_crop_id) if progress.last_crop_id else None
        except (ValueError, TypeError, AttributeError):
            crop_id = None
        return created_at, crop_id

    @classmethod
    def _advance_cursor(
        cls,
        progress: AttributeBackfillProgress,
        crop: PersonCrop,
    ) -> tuple[datetime | None, uuid.UUID | None]:
        crop_id = str(crop.id)
        progress.last_crop_id = crop_id
        created_at = getattr(crop, "created_at", None)
        if not isinstance(created_at, datetime):
            # Legacy test doubles and hand-created rows may not expose created_at.  Their source
            # predicate still prevents duplicate work; simply leave the timestamp cursor absent.
            return None, None
        progress.last_created_at = created_at.isoformat()
        try:
            return created_at, uuid.UUID(crop_id)
        except (ValueError, TypeError, AttributeError):
            return created_at, None

    def _save(self, progress: AttributeBackfillProgress) -> None:
        progress.updated_at = datetime.now(UTC).isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(progress), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)
