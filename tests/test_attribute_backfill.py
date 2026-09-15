import uuid
from datetime import UTC, datetime, timedelta

from app.config.settings import Settings
from app.services.attribute_backfill import DurableAttributeBackfillService


class _Query:
    def __init__(self, db):
        self.db = db
        self._limit = None
        self._offset = 0

    def order_by(self, *_columns):
        return self

    def limit(self, value):
        self.db.limit_calls.append(value)
        self._limit = value
        return self

    def offset(self, value):
        self._offset = value
        return self

    def all(self):
        assert self._limit is not None, "backfill must never issue an unbounded all()"
        rows = sorted(
            self.db.rows,
            key=lambda row: (
                getattr(row, "created_at", None) or datetime.min,
                str(row.id),
            ),
        )
        return rows[self._offset : self._offset + self._limit]


class _Db:
    def __init__(self, rows):
        self.rows = rows
        self.limit_calls = []

    def query(self, _model):
        return _Query(self)


def _crop(source, *, created_at=None):
    return type(
        "Crop",
        (),
        {
            "id": uuid.uuid4(),
            "attributes": {"source": source} if source else None,
            "created_at": created_at,
        },
    )()


def test_durable_backfill_resumes_and_quarantines_repeated_failures(monkeypatch, tmp_path):
    described = _crop("vlm")
    tone = _crop("cv_tone")
    broken = _crop(None)
    state_path = tmp_path / "attribute-backfill.json"
    service = DurableAttributeBackfillService(
        _Db([described, tone, broken]),
        Settings(data_dir=tmp_path),
        state_path=state_path,
    )

    def analyze(crop, *, persist):
        assert persist is True
        if crop is broken:
            raise ValueError("unreadable image")
        crop.attributes = {"source": "vlm"}

    monkeypatch.setattr(service.structured, "analyze_person_crop", analyze)

    progress = service.run(batch_size=2, max_attempts=2)

    assert progress.completed is True
    assert progress.attempted == 3
    assert progress.updated == 1
    assert progress.permanent_failures == {str(broken.id): "unreadable image"}
    assert state_path.is_file()

    resumed = DurableAttributeBackfillService(
        _Db([described, tone, broken]),
        Settings(data_dir=tmp_path),
        state_path=state_path,
    ).run(batch_size=2, max_attempts=2)

    assert resumed.attempted == 3
    assert resumed.updated == 1


def test_backfill_uses_bounded_ascending_pages_and_does_not_starve_old_rows(monkeypatch, tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    oldest = _crop(None, created_at=start)
    existing_newer = _crop(None, created_at=start + timedelta(seconds=1))
    rows = [oldest, existing_newer]
    db = _Db(rows)
    service = DurableAttributeBackfillService(
        db,
        Settings(data_dir=tmp_path),
        state_path=tmp_path / "attribute-backfill.json",
    )
    seen = []
    inserted = 0

    def analyze(crop, *, persist):
        nonlocal inserted
        assert persist is True
        seen.append(crop)
        crop.attributes = {"source": "vlm"}
        # Simulate a live camera continuously appending newer crops while the old backlog is
        # being consumed.  The oldest row must still be selected first and completed.
        if inserted < 4:
            rows.append(
                _crop(
                    None,
                    created_at=start + timedelta(seconds=10 + inserted),
                )
            )
            inserted += 1

    monkeypatch.setattr(service.structured, "analyze_person_crop", analyze)

    progress = service.run(batch_size=1)

    assert progress.completed is True
    assert seen[0] is oldest
    assert oldest.attributes == {"source": "vlm"}
    assert all(crop.attributes == {"source": "vlm"} for crop in rows)
    assert db.limit_calls and max(db.limit_calls) <= 1
    assert progress.last_created_at is not None


def test_custom_backfill_processor_keeps_legacy_batch_and_status_counts(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    described = _crop("vlm", created_at=start)
    unreadable = _crop(None, created_at=start + timedelta(seconds=1))
    fresh = _crop(None, created_at=start + timedelta(seconds=2))
    db = _Db([described, unreadable, fresh])
    state_path = tmp_path / "tone-backfill.json"
    service = DurableAttributeBackfillService(
        db,
        Settings(data_dir=tmp_path),
        state_path=state_path,
    )

    def process(crop):
        if crop is described:
            return "skipped"
        if crop is unreadable:
            return "unreadable"
        crop.attributes = {"source": "cv_tone"}
        return "updated"

    first = service.run(
        batch_size=2,
        processor=process,
        include_described=True,
        max_items=2,
    )

    assert (first.attempted, first.updated, first.skipped, first.unreadable) == (2, 0, 1, 1)

    second = DurableAttributeBackfillService(
        db,
        Settings(data_dir=tmp_path),
        state_path=state_path,
    ).run(
        batch_size=2,
        processor=process,
        include_described=True,
        max_items=2,
    )
    assert (second.attempted, second.updated, second.skipped, second.unreadable) == (3, 1, 1, 1)


def test_vlm_backfill_http_compatibility_route_resumes_from_checkpoint(monkeypatch, tmp_path):
    """The legacy HTTP entry point must use the durable walker, not an unbounded query.all()."""

    from fastapi.testclient import TestClient
    from test_reid import load_app

    main = load_app(monkeypatch, tmp_path, "test-vlm-backfill-http")
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.structured_attributes import StructuredAttributeService

    calls = []

    def analyze(_service, crop, *, persist=True):
        assert persist is True
        calls.append(crop.id)
        crop.attributes = {"source": "vlm", "clothing": {"upper_color": "black"}}
        return crop.attributes

    monkeypatch.setattr(StructuredAttributeService, "analyze_person_crop", analyze)

    with TestClient(main.create_app()) as client:
        from app.config.settings import get_settings

        with SessionLocal() as db:
            image = Image(image_url="/data/frame.jpg", source_type="stream_frame")
            db.add(image)
            db.flush()
            crops = [
                PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/{index}.jpg",
                    bbox={"label": "person"},
                )
                for index in range(3)
            ]
            db.add_all(crops)
            db.commit()

        first = client.post("/api/attributes/person-crops/backfill?limit=2").json()
        second = client.post("/api/attributes/person-crops/backfill?limit=2").json()

        state_path = get_settings().data_dir / "tasks" / "attribute-backfill.json"

    assert (first["seen"], first["updated"], first["errors"]) == (2, 2, [])
    assert (second["seen"], second["updated"], second["errors"]) == (1, 1, [])
    assert len(calls) == 3
    assert state_path.is_file()
