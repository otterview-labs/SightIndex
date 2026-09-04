import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.models.media import PersonCrop
from app.services.structured_attributes import StructuredAttributeService
from app.services.vlm import VLMRuntimeError

logger = logging.getLogger(__name__)


@dataclass
class AttributeBackfillProgress:
    attempted: int = 0
    updated: int = 0
    remaining: int = 0
    completed: bool = False
    last_crop_id: str | None = None
    failures: dict[str, int] = field(default_factory=dict)
    permanent_failures: dict[str, str] = field(default_factory=dict)
    updated_at: str | None = None


class DurableAttributeBackfillService:
    """Upgrade legacy crop tags with a durable, restart-safe progress checkpoint."""

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
        self.state_path = state_path or (
            settings.data_dir / "tasks" / "attribute-backfill.json"
        )

    def run(self, *, batch_size: int = 50, max_attempts: int = 3) -> AttributeBackfillProgress:
        progress = self.load_progress()
        progress.completed = False
        while True:
            candidates, remaining = self._pending(
                batch_size,
                excluded=set(progress.permanent_failures),
            )
            progress.remaining = remaining
            if not candidates:
                progress.completed = remaining == 0
                self._save(progress)
                return progress
            for crop in candidates:
                crop_id = str(crop.id)
                progress.attempted += 1
                progress.last_crop_id = crop_id
                try:
                    self.structured.analyze_person_crop(crop, persist=True)
                except (VLMRuntimeError, OSError, ValueError) as exc:
                    attempts = progress.failures.get(crop_id, 0) + 1
                    progress.failures[crop_id] = attempts
                    if attempts >= max_attempts:
                        progress.permanent_failures[crop_id] = str(exc)
                    logger.warning(
                        "Attribute backfill failed for %s (attempt %s/%s): %s",
                        crop_id,
                        attempts,
                        max_attempts,
                        exc,
                    )
                else:
                    progress.updated += 1
                    progress.failures.pop(crop_id, None)
                self._save(progress)

    def load_progress(self) -> AttributeBackfillProgress:
        if not self.state_path.is_file():
            return AttributeBackfillProgress()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return AttributeBackfillProgress(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable attribute backfill state", exc_info=True)
            return AttributeBackfillProgress()

    def _pending(
        self,
        limit: int,
        *,
        excluded: set[str],
    ) -> tuple[list[PersonCrop], int]:
        rows = self.db.query(PersonCrop).order_by(PersonCrop.created_at.desc()).all()
        pending = [
            crop
            for crop in rows
            if str(crop.id) not in excluded
            and (
                not isinstance(crop.attributes, dict)
                or crop.attributes.get("source") != "vlm"
            )
        ]
        return pending[:limit], len(pending)

    def _save(self, progress: AttributeBackfillProgress) -> None:
        progress.updated_at = datetime.now(UTC).isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(progress), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)
