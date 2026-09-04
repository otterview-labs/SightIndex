import uuid

from app.config.settings import Settings
from app.services.attribute_backfill import DurableAttributeBackfillService


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def order_by(self, _column):
        return self

    def all(self):
        return self.rows


class _Db:
    def __init__(self, rows):
        self.rows = rows

    def query(self, _model):
        return _Query(self.rows)


def _crop(source):
    return type(
        "Crop",
        (),
        {
            "id": uuid.uuid4(),
            "attributes": {"source": source} if source else None,
        },
    )()


def test_durable_backfill_resumes_and_quarantines_repeated_failures(
    monkeypatch, tmp_path
):
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
