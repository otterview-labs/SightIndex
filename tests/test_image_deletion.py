"""DELETE /api/images/{image_id}: subtree removal, idempotency and backend refusal."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from test_reid import load_app


class FakeCollection:
    """Records the delete expressions a real Milvus collection would receive."""

    def __init__(self, object_type: str, calls: list[tuple[str, str]]) -> None:
        self.object_type = object_type
        self.calls = calls

    def delete(self, expr: str | None = None) -> None:
        self.calls.append((self.object_type, expr or ""))

    def flush(self) -> None:
        return None


def install_fake_milvus(monkeypatch, available: bool = True) -> list[tuple[str, str]]:
    from app.services.vector_index import MilvusVectorIndex

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        MilvusVectorIndex,
        "probe",
        lambda self, timeout_seconds=2.0: (
            available,
            None if available else "connection refused",
        ),
    )
    monkeypatch.setattr(
        MilvusVectorIndex,
        "_collection",
        lambda self, object_type: FakeCollection(object_type, calls),
    )
    return calls


def build_state(monkeypatch, tmp_path, name: str, **env: str):
    main = load_app(
        monkeypatch,
        tmp_path,
        name,
        VECTOR_INDEX_ON_INGEST="false",
        REID_ENABLED="false",
        VLM_PROVIDER="none",
        **env,
    )

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop, PersonObservationIndex
    from app.models.persons import Person
    from app.models.vectors import FaceEmbedding, VectorIndexJob, VLEmbedding

    settings = get_settings()

    def seed(crops: int = 2, enrolled_face: bool = False, files: bool = True):
        image_id = uuid.uuid4()
        crop_ids = [uuid.uuid4() for _ in range(crops)]
        image_name = f"{image_id}.png"
        with SessionLocal() as db:
            db.add(
                Image(
                    id=image_id,
                    image_url=f"/data/uploads/{image_name}",
                    thumbnail_url=f"/data/thumbnails/{image_name}",
                    source_type="upload",
                )
            )
            for crop_id in crop_ids:
                db.add(
                    PersonCrop(
                        id=crop_id,
                        image_id=image_id,
                        crop_url=f"/data/crops/{crop_id}.png",
                        bbox={"x": 1, "y": 2, "w": 3, "h": 4},
                    )
                )
                db.add(
                    PersonObservationIndex(
                        id=uuid.uuid4(), crop_id=crop_id, image_id=image_id
                    )
                )
                db.add(
                    VLEmbedding(
                        id=uuid.uuid4(),
                        object_type="person_crop",
                        object_id=crop_id,
                        embedding_model="test-model",
                        embedding_dim=4,
                    )
                )
                db.add(VectorIndexJob(id=uuid.uuid4(), target="person_crop", object_id=crop_id))
            if enrolled_face:
                person = Person(id=uuid.uuid4(), name="张三")
                db.add(person)
                db.add(
                    FaceEmbedding(
                        id=uuid.uuid4(),
                        person_id=person.id,
                        image_id=image_id,
                        face_model="test-model",
                    )
                )
            db.commit()

        urls = [
            f"/data/uploads/{image_name}",
            f"/data/thumbnails/{image_name}",
            *(f"/data/crops/{crop_id}.png" for crop_id in crop_ids),
        ]
        paths = []
        for url in urls:
            path = settings.data_dir / url.removeprefix("/data/")
            if files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"payload")
            paths.append(path)
        return SimpleNamespace(image_id=image_id, crop_ids=crop_ids, paths=paths, urls=urls)

    def count(model) -> int:
        with SessionLocal() as db:
            return db.scalar(select(func.count()).select_from(model))

    return SimpleNamespace(
        main=main,
        seed=seed,
        count=count,
        db=SessionLocal,
        settings=settings,
        Image=Image,
        PersonCrop=PersonCrop,
        PersonObservationIndex=PersonObservationIndex,
        VLEmbedding=VLEmbedding,
        VectorIndexJob=VectorIndexJob,
        FaceEmbedding=FaceEmbedding,
    )


@pytest.fixture
def deletion_app(monkeypatch, tmp_path):
    state = build_state(monkeypatch, tmp_path, "image-deletion", MILVUS_ENABLED="false")
    with TestClient(state.main.create_app()) as client:
        state.client = client
        yield state


@pytest.fixture
def deletion_app_with_milvus(monkeypatch, tmp_path):
    state = build_state(monkeypatch, tmp_path, "image-deletion-milvus", MILVUS_ENABLED="true")
    with TestClient(state.main.create_app()) as client:
        state.client = client
        yield state


def test_dry_run_counts_the_subtree_without_writing(deletion_app):
    state = deletion_app
    seeded = state.seed()

    response = state.client.request(
        "DELETE", f"/api/images/{seeded.image_id}", params={"dry_run": "true"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dry_run"] is True
    assert payload["deleted"]["images"] == 1
    assert payload["deleted"]["person_crops"] == 2
    assert payload["deleted"]["person_observation_index"] == 2
    assert payload["deleted"]["vl_embeddings"] == 2
    assert payload["deleted"]["vector_index_jobs"] == 2
    assert payload["deleted"]["files"] == 4

    # A dry run must not touch the database or the disk.
    with state.db() as db:
        assert db.get(state.Image, seeded.image_id) is not None
    assert state.count(state.PersonCrop) == 2
    assert state.count(state.PersonObservationIndex) == 2
    assert all(path.exists() for path in seeded.paths)


def test_delete_removes_rows_and_files(deletion_app):
    state = deletion_app
    seeded = state.seed()

    response = state.client.request("DELETE", f"/api/images/{seeded.image_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["dry_run"] is False
    assert payload["deleted"]["person_crops"] == 2
    assert payload["deleted"]["files"] == 4

    assert state.count(state.Image) == 0
    assert state.count(state.PersonCrop) == 0
    assert state.count(state.PersonObservationIndex) == 0
    assert state.count(state.VLEmbedding) == 0
    assert state.count(state.VectorIndexJob) == 0
    assert not any(path.exists() for path in seeded.paths)


def test_observations_are_cleared_before_their_crops(deletion_app):
    """person_observation_index holds a NO ACTION reference, so the order is load bearing."""

    state = deletion_app
    seeded = state.seed(crops=3)

    response = state.client.request("DELETE", f"/api/images/{seeded.image_id}")

    assert response.status_code == 200
    assert response.json()["deleted"]["person_observation_index"] == 3
    assert state.count(state.PersonObservationIndex) == 0
    assert state.count(state.PersonCrop) == 0


def test_delete_without_files_keeps_them_on_disk(deletion_app):
    state = deletion_app
    seeded = state.seed()

    response = state.client.request(
        "DELETE", f"/api/images/{seeded.image_id}", params={"delete_files": "false"}
    )

    assert response.status_code == 200
    assert response.json()["deleted"]["files"] == 0
    assert state.count(state.Image) == 0
    assert all(path.exists() for path in seeded.paths)


def test_deleting_a_missing_image_is_idempotent(deletion_app):
    state = deletion_app
    seeded = state.seed()

    first = state.client.request("DELETE", f"/api/images/{seeded.image_id}")
    assert first.status_code == 200

    second = state.client.request("DELETE", f"/api/images/{seeded.image_id}")
    assert second.status_code == 404

    unknown = state.client.request("DELETE", f"/api/images/{uuid.uuid4()}")
    assert unknown.status_code == 404


def test_enrolled_face_removal_is_reported_as_a_warning(deletion_app):
    state = deletion_app
    seeded = state.seed(crops=1, enrolled_face=True)

    response = state.client.request(
        "DELETE", f"/api/images/{seeded.image_id}", params={"dry_run": "true"}
    )

    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert len(warnings) == 1
    assert "已登记人员" in warnings[0]


def test_nothing_is_destroyed_when_milvus_is_unreachable(deletion_app_with_milvus, monkeypatch):
    state = deletion_app_with_milvus
    install_fake_milvus(monkeypatch, available=False)
    seeded = state.seed()

    response = state.client.request("DELETE", f"/api/images/{seeded.image_id}")

    assert response.status_code == 503
    assert "Milvus" in response.json()["detail"]
    # The refusal has to happen before any row or file is touched.
    assert state.count(state.Image) == 1
    assert state.count(state.PersonCrop) == 2
    assert all(path.exists() for path in seeded.paths)


def test_vectors_are_deleted_from_all_three_collections(deletion_app_with_milvus, monkeypatch):
    state = deletion_app_with_milvus
    calls = install_fake_milvus(monkeypatch, available=True)
    seeded = state.seed()

    response = state.client.request("DELETE", f"/api/images/{seeded.image_id}")

    assert response.status_code == 200
    payload = response.json()["deleted"]["milvus"]
    assert payload == {"image": 1, "person_crop": 2, "reid_person_crop": 2}

    touched = [object_type for object_type, _expr in calls]
    assert touched == ["image", "person_crop", "reid_person_crop"]

    expressions = dict(calls)
    assert str(seeded.image_id) in expressions["image"]
    for crop_id in seeded.crop_ids:
        assert str(crop_id) in expressions["person_crop"]
        assert str(crop_id) in expressions["reid_person_crop"]
