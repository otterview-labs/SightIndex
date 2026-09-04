from app.config.settings import Settings
from app.services.vector_index import VectorIndexingService


class _FakeDB:
    def __init__(self, items: list[object]) -> None:
        self.items = items
        self.commits = 0
        self.rollbacks = 0

    def scalars(self, _statement: object) -> list[object]:
        return self.items

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _FakeIndex:
    def __init__(self) -> None:
        self.flushes: list[str] = []

    def flush(self, target: str) -> None:
        self.flushes.append(target)


def test_rebuild_flushes_and_commits_in_bounded_batches() -> None:
    service = VectorIndexingService.__new__(VectorIndexingService)
    service.db = _FakeDB([object() for _ in range(205)])
    service.settings = Settings()
    service.index = _FakeIndex()
    service.index_crop = lambda _item, *, flush: None

    result = service.rebuild("person_crop", 205)

    assert result == {
        "target": "person_crop",
        "requested": 205,
        "seen": 205,
        "indexed": 205,
        "errors": [],
    }
    assert service.index.flushes == ["person_crop", "person_crop", "person_crop"]
    assert service.db.commits == 205
    assert service.db.rollbacks == 0
