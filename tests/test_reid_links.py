"""Per-camera best candidate: ranking, not a verdict."""

import uuid
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from test_reid import load_app


@pytest.fixture
def wired(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "test-reid-links",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop, VideoStream
    from app.services.observation_index import ObservationIndexService
    from app.services.reid import ReidEmbeddingService
    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    doors = {"产品部门口": uuid.uuid4(), "项目部门口": uuid.uuid4(), "食堂": uuid.uuid4()}
    ids: dict[str, uuid.UUID] = {}
    scores: dict[str, float] = {}

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)
    monkeypatch.setattr(MilvusVectorIndex, "fetch_vectors", lambda self, *args: {})
    monkeypatch.setattr(ReidEmbeddingService, "embed_image", lambda self, path: [0.1] * self.dim)
    monkeypatch.setattr(
        MilvusVectorIndex,
        "search_vector",
        lambda self, object_type, vector, top_k: [
            VectorSearchHit(object_id=ids[name], score=score) for name, score in scores.items()
        ],
    )

    with TestClient(main.create_app()) as client:
        crops_dir = get_settings().data_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        with SessionLocal() as db:
            for name, camera_id in doors.items():
                db.add(VideoStream(name=name, stream_url="rtsp://x", camera_id=camera_id))
            image = Image(image_url="/data/frames/f.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)

            def add(key, camera, minute, attributes=None):
                (crops_dir / f"{key}.jpg").write_bytes(b"bytes")
                crop = PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/{key}.jpg",
                    bbox={"label": "person"},
                    camera_id=doors[camera],
                    captured_at=datetime(2026, 8, 24, 12, minute, 0),
                    attributes=attributes,
                )
                db.add(crop)
                db.commit()
                db.refresh(crop)
                ids[key] = crop.id
                ObservationIndexService(db, get_settings()).upsert_crop(crop)

            black_clothes = {
                "clothing": {
                    "upper_color": "black",
                    "lower_color": "black",
                    "upper_color_confidence": 0.95,
                    "lower_color_confidence": 0.92,
                }
            }
            add("query", "产品部门口", 0, black_clothes)
            add("home_strong", "产品部门口", 1)
            add("b_weak", "项目部门口", 2)
            add("b_best", "项目部门口", 3, black_clothes)
            add("c_faint", "食堂", 4)
            db.commit()
        scores.update(
            {"query": 1.0, "home_strong": 0.91, "b_weak": 0.30, "b_best": 0.47, "c_faint": 0.21}
        )
        yield client, ids, doors


def test_each_other_camera_contributes_its_best(wired):
    client, ids, doors = wired

    payload = client.post(f"/api/reid/crops/{ids['query']}/links").json()

    assert [link["camera_name"] for link in payload["links"]] == ["项目部门口", "食堂"]
    assert payload["links"][0]["crop_id"] == str(ids["b_best"]), "the weaker one at that door won"


def test_a_faint_link_is_still_returned_but_marked(wired):
    """No threshold hides it; the flag says whether coincidence could explain it."""

    client, ids, _ = wired

    links = client.post(f"/api/reid/crops/{ids['query']}/links").json()["links"]
    by_camera = {link["camera_name"]: link for link in links}

    assert by_camera["项目部门口"]["beats_chance"] is True
    assert by_camera["食堂"]["beats_chance"] is False, "0.21 is well inside coincidence"


def test_the_camera_searched_from_is_not_a_link(wired):
    """The question is where else they went, not that they were where we already know."""

    client, ids, _ = wired

    links = client.post(f"/api/reid/crops/{ids['query']}/links").json()["links"]

    assert "产品部门口" not in {link["camera_name"] for link in links}
    assert str(ids["query"]) not in {link["crop_id"] for link in links}


def test_the_response_says_where_the_query_came_from(wired):
    client, ids, _ = wired

    payload = client.post(f"/api/reid/crops/{ids['query']}/links").json()

    assert payload["camera_name"] == "产品部门口"
    assert payload["chance_ceiling"] == 0.44


def test_a_labelled_candidate_carries_its_name_into_the_link(monkeypatch, tmp_path):
    """ReidCameraLink has its own field list, separate from ReidMatchItem's: a name attached to
    the same underlying item is not free just because /similar already carries it."""

    main = load_app(
        monkeypatch,
        tmp_path,
        "test-reid-link-name",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop, VideoStream
    from app.models.persons import Person
    from app.services.observation_index import ObservationIndexService
    from app.services.reid import ReidEmbeddingService
    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    query_camera = uuid.uuid4()
    other_camera = uuid.uuid4()
    ids: dict[str, uuid.UUID] = {}

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)
    monkeypatch.setattr(MilvusVectorIndex, "fetch_vectors", lambda self, *args: {})
    monkeypatch.setattr(ReidEmbeddingService, "embed_image", lambda self, path: [0.1] * self.dim)
    monkeypatch.setattr(
        MilvusVectorIndex,
        "search_vector",
        lambda self, object_type, vector, top_k: [
            VectorSearchHit(object_id=ids["query"], score=1.0),
            VectorSearchHit(object_id=ids["labelled"], score=0.9),
        ],
    )

    with TestClient(main.create_app()) as client:
        crops_dir = get_settings().data_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        with SessionLocal() as db:
            db.add(VideoStream(name="产品部门口", stream_url="rtsp://x", camera_id=query_camera))
            db.add(VideoStream(name="项目部门口", stream_url="rtsp://x", camera_id=other_camera))
            image = Image(image_url="/data/frames/f.jpg", source_type="stream_frame")
            db.add(image)
            person = Person(name="王五")
            db.add(person)
            db.commit()
            db.refresh(image)
            db.refresh(person)
            person_id = person.id

            def add(key, camera_id, person_id=None):
                (crops_dir / f"{key}.jpg").write_bytes(b"bytes")
                crop = PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/{key}.jpg",
                    bbox={"label": "person"},
                    camera_id=camera_id,
                    captured_at=datetime(2026, 8, 24, 12, 0, 0),
                    person_id=person_id,
                )
                db.add(crop)
                db.commit()
                db.refresh(crop)
                ids[key] = crop.id
                ObservationIndexService(db, get_settings()).upsert_crop(crop)

            add("query", query_camera)
            add("labelled", other_camera, person_id=person_id)
            db.commit()

        payload = client.post(f"/api/reid/crops/{ids['query']}/links").json()

    assert len(payload["links"]) == 1
    link = payload["links"][0]
    assert link["person_id"] == str(person_id)
    assert link["person_name"] == "王五"
    assert link["person_is_vip"] is False


def test_camera_link_reports_when_global_pool_may_omit_cameras(wired, monkeypatch):
    """The diagnostic exposes global-pool truncation without changing link ranking."""

    from app.services.reid_index import ReidIndexService

    # The fixture's fake search deliberately returns all five rows even when the requested pool
    # is three.  This lets the response test observe the counters without a live Milvus server.
    monkeypatch.setattr(ReidIndexService, "candidate_pool_limit", lambda self: 3)
    monkeypatch.setattr(
        ReidIndexService,
        "indexed_camera_count",
        lambda self, **_kwargs: 3,
    )
    client, ids, _ = wired

    payload = client.post(f"/api/reid/crops/{ids['query']}/links").json()
    coverage = payload["candidate_coverage"]

    assert coverage == {
        "raw_hit_count": 5,
        "hit_camera_count": 2,  # source camera is intentionally excluded
        "indexed_camera_count": 3,
        "pool_limit": 3,
        "sql_row_missing_count": 0,
        "possibly_truncated": True,
    }
    # Coverage is explanatory only; the existing per-camera result remains unchanged.
    assert [link["camera_name"] for link in payload["links"]] == ["项目部门口", "食堂"]


def test_camera_link_reports_stale_milvus_hits_separately(wired, monkeypatch):
    """A vector without a SQL crop is a consistency gap, not a negative identity result."""

    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    missing_crop_id = uuid.uuid4()
    monkeypatch.setattr(
        MilvusVectorIndex,
        "search_vector",
        lambda self, _object_type, _vector, _top_k: [
            VectorSearchHit(object_id=missing_crop_id, score=0.88)
        ],
    )
    client, ids, _ = wired

    payload = client.post(f"/api/reid/crops/{ids['query']}/links").json()

    assert payload["links"] == []
    assert payload["candidate_coverage"]["raw_hit_count"] == 1
    assert payload["candidate_coverage"]["sql_row_missing_count"] == 1
    assert payload["candidate_coverage"]["possibly_truncated"] is False


def test_camera_link_keeps_explainable_attribute_counts(wired):
    """The link DTO must not discard counts already calculated for its label explanation."""

    client, ids, _ = wired

    links = client.post(f"/api/reid/crops/{ids['query']}/links").json()["links"]
    project_door = next(link for link in links if link["camera_name"] == "项目部门口")

    assert project_door["attribute_agreement"] == 1.0
    assert project_door["attribute_comparable_count"] == 2
    assert project_door["attribute_match_count"] == 2
    assert project_door["attribute_conflict_count"] == 0


def test_an_unknown_crop_is_a_404(wired):
    client, _, _ = wired

    assert client.post(f"/api/reid/crops/{uuid.uuid4()}/links").status_code == 404


def test_camera_link_rejects_strong_face_conflict_and_uses_next_candidate(wired, monkeypatch):
    from types import SimpleNamespace

    from app.schemas.reid import ReidFaceCoverage

    client, ids, _ = wired

    def conflicting_face(db, settings, crop, items, **kwargs):
        for item in items:
            if item.crop_id == ids["b_best"]:
                item.face_similarity = 0.1
                item.face_match = False
                item.face_reliability = 0.95

    monkeypatch.setattr("app.api.reid.enrich_camera_link_face_evidence", conflicting_face)
    monkeypatch.setattr(
        "app.api.reid.prepare_camera_link_face_query",
        lambda *args, **kwargs: SimpleNamespace(face=object(), coverage=ReidFaceCoverage()),
    )
    response = client.post(f"/api/reid/crops/{ids['query']}/links")
    assert response.status_code == 200
    links = response.json()["links"]
    assert str(ids["b_best"]) not in {link["crop_id"] for link in links}
    assert str(ids["b_weak"]) in {link["crop_id"] for link in links}
