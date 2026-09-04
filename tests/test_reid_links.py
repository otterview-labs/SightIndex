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
    monkeypatch.setattr(
        ReidEmbeddingService, "embed_image", lambda self, path: [0.1] * self.dim
    )
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
