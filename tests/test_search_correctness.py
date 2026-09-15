import importlib
import sys
import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'search.db'}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PERSON_DETECTOR", "whole_frame")
    monkeypatch.setenv("VLM_PROVIDER", "none")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "none")
    monkeypatch.setenv("REID_ENABLED", "false")
    monkeypatch.setenv("LOCAL_TIMEZONE", "Asia/Shanghai")
    for name in list(sys.modules):
        if name == "main" or name.startswith("app."):
            sys.modules.pop(name)
    main = importlib.import_module("main")
    with TestClient(main.create_app()) as instance:
        yield instance


def seed(attributes, *, minutes=0, camera_id=None, person_id=None, label="person"):
    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.observation_index import ObservationIndexService

    with SessionLocal() as db:
        captured = datetime(2026, 9, 15, 9, 35) + timedelta(minutes=minutes)
        image = Image(
            image_url="/data/frames/synthetic.png",
            source_type="upload",
            captured_at=captured,
            camera_id=camera_id,
        )
        db.add(image)
        db.flush()
        crop = PersonCrop(
            image_id=image.id,
            crop_url=f"/data/crops/{uuid.uuid4()}.png",
            bbox={"label": label},
            attributes=attributes,
            captured_at=captured,
            created_at=captured,
            camera_id=camera_id,
            person_id=person_id,
        )
        db.add(crop)
        db.flush()
        ObservationIndexService(db, get_settings()).upsert_crop(crop)
        db.commit()
        return str(crop.id)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2026-09-15T09:30:00", "2026-09-15T09:40:00"),
        ("2026-09-15T09:30:00+08:00", "2026-09-15T09:40:00+08:00"),
        ("2026-09-15T01:30:00Z", "2026-09-15T01:40:00Z"),
    ],
)
def test_same_time_interval_returns_same_crops(client, start, end):
    matching = seed({"clothing": {"upper_color": "black"}, "objects": {"backpack": True}})
    seed({"clothing": {"upper_color": "black"}, "objects": {"backpack": True}}, minutes=20)
    filters = {"start_time": start, "end_time": end}
    response = client.post(
        "/api/search/person-crops", json={"query": "黑衣背包的人", "filters": filters}
    )
    assert response.status_code == 200, response.text
    assert [item["crop_id"] for item in response.json()["items"]] == [matching]
    observation = client.get("/api/search/observations", params={"query": "黑衣背包", **filters})
    assert observation.status_code == 200, observation.text
    assert observation.json()["total"] == 1
    assert [item["crop_id"] for item in observation.json()["items"]] == [matching]


@pytest.mark.parametrize(
    ("query", "field", "group"),
    [
        ("不戴眼镜的人", "glasses", "appearance"),
        ("没有戴眼镜", "glasses", "appearance"),
        ("不戴帽子的人", "hat", "appearance"),
        ("没背包的人", "backpack", "objects"),
        ("没有背包的人", "backpack", "objects"),
        ("不拿手机的人", "holding_phone", "objects"),
        ("不看手机的人", "holding_phone", "objects"),
        ("不抽烟的人", "smoking", "behavior"),
        ("没跌倒的人", "falling", "behavior"),
        ("没有打架的人", "fighting", "behavior"),
        ("person without glasses", "glasses", "appearance"),
        ("person not wearing glasses", "glasses", "appearance"),
        ("person without a backpack", "backpack", "objects"),
    ],
)
def test_negative_conditions_require_explicit_false(client, query, field, group):
    matching = seed({group: {field: False}})
    seed({group: {field: True}})
    seed({group: {field: None}})
    seed({group: {field: "unknown"}})
    response = client.post("/api/search/person-crops", json={"query": query})
    assert response.status_code == 200, response.text
    assert [item["crop_id"] for item in response.json()["items"]] == [matching]
    observation = client.get("/api/search/observations", params={"query": query})
    assert observation.status_code == 200, observation.text
    assert observation.json()["total"] == 1
    assert [item["crop_id"] for item in observation.json()["items"]] == [matching]


def test_negation_is_scoped_to_the_correct_attribute(client):
    expected = seed({"appearance": {"glasses": False}, "objects": {"backpack": True}})
    seed({"appearance": {"glasses": True}, "objects": {"backpack": True}})
    seed({"appearance": {"glasses": False}, "objects": {"backpack": False}})
    response = client.post(
        "/api/search/person-crops", json={"query": "不戴眼镜，背包的人"}
    )
    assert response.status_code == 200, response.text
    assert [item["crop_id"] for item in response.json()["items"]] == [expected]


def test_negative_search_excludes_conflicting_detector_evidence(client):
    seed({"objects": {"holding_phone": False}}, label="phone")
    seed({"behavior": {"smoking": False}, "objects": {"cigarette": True}})
    for query in ("不拿手机的人", "不抽烟的人"):
        response = client.post("/api/search/person-crops", json={"query": query})
        assert response.status_code == 200, response.text
        assert response.json()["items"] == []


@pytest.mark.parametrize(
    ("attributes", "label", "query"),
    [
        ({"backpack": "\u3000YES\u3000"}, "person", "背包"),
        ({"has_backpack": " 是 "}, "person", "背包"),
        ({"backpack": " FALSE "}, "person", "没有背包"),
        ({"has_backpack": "\u3000否\u3000"}, "person", "没有背包"),
        ({}, "phone", "拿手机"),
        ({"behavior": {"fallen": True}}, "person", "跌倒"),
        ({"objects": {"cigarette": "true"}}, "person", "抽烟"),
    ],
)
def test_sql_prefilter_keeps_legacy_aliases_and_detector_evidence(client, attributes, label, query):
    matching = seed(attributes, label=label)
    response = client.post("/api/search/person-crops", json={"query": query})
    assert response.status_code == 200, response.text
    assert [item["crop_id"] for item in response.json()["items"]] == [matching]
    response = client.get("/api/search/observations", params={"query": query})
    assert response.status_code == 200, response.text
    assert [item["crop_id"] for item in response.json()["items"]] == [matching]


def test_observation_keyword_uses_values_and_paginates_matches(client):
    matching = [seed({"objects": {"backpack": True}}, minutes=index) for index in range(3)]
    seed({"objects": {"backpack": False}}, minutes=4)
    seed({"objects": {"backpack": None}}, minutes=5)
    response = client.get(
        "/api/search/observations", params={"query": "背包", "offset": 1, "limit": 1}
    )
    assert response.status_code == 200, response.text
    assert response.json()["total"] == 3
    assert [item["crop_id"] for item in response.json()["items"]] == [matching[1]]


def test_plain_person_name_query_remains_available(client):
    from app.db.session import SessionLocal
    from app.models.persons import Person

    with SessionLocal() as db:
        person = Person(name="验收样本人员")
        db.add(person)
        db.commit()
        person_id = person.id
    expected = seed({"objects": {"backpack": None}}, person_id=person_id)
    response = client.get("/api/search/observations", params={"query": "验收样本人员"})
    assert response.status_code == 200
    assert [item["crop_id"] for item in response.json()["items"]] == [expected]


def test_person_name_does_not_bypass_negative_constraints(client):
    from app.db.session import SessionLocal
    from app.models.persons import Person

    with SessionLocal() as db:
        person = Person(name="验收样本人员")
        db.add(person)
        db.commit()
        person_id = person.id
    expected = seed({"appearance": {"glasses": False}}, person_id=person_id)
    seed({"appearance": {"glasses": True}}, person_id=person_id)
    seed({"appearance": {"glasses": False}})
    response = client.post(
        "/api/search/person-crops", json={"query": "验收样本人员不戴眼镜"}
    )
    assert response.status_code == 200, response.text
    assert [item["crop_id"] for item in response.json()["items"]] == [expected]


@pytest.mark.parametrize("query", ["不是黑色上衣", "戴眼镜或背包", "戴眼镜且不戴眼镜", "眼镜未知"])
def test_unsupported_or_contradictory_constraints_are_not_silently_dropped(client, query):
    seed({"clothing": {"upper_color": "black"}, "appearance": {"glasses": True}})
    response = client.post("/api/search/person-crops", json={"query": query})
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str)
    observation = client.get("/api/search/observations", params={"query": query})
    assert observation.status_code == 400


def test_invalid_time_range_returns_clear_client_error(client):
    filters = {"start_time": "2026-09-15T10:00:00+08:00", "end_time": "2026-09-15T01:00:00Z"}
    response = client.post(
        "/api/search/person-crops", json={"query": "背包", "filters": filters}
    )
    assert response.status_code == 400
    assert "开始时间" in response.json()["detail"]
    observation = client.get("/api/search/observations", params=filters)
    assert observation.status_code == 400


def test_requested_face_diagnostics_never_fall_back_to_recent_crops(client, monkeypatch):
    from app.services.faces import FaceRecognitionService

    calls = []

    def no_face(self, url, allow_fallback):
        calls.append(url)
        return None

    monkeypatch.setattr(FaceRecognitionService, "_best_candidate", no_face)
    oldest = seed({}, minutes=-200)
    newest = seed({})
    response = client.post(
        "/api/face/diagnostics/crops", json={"crop_ids": [oldest, newest, oldest]}
    )
    assert response.status_code == 200, response.text
    assert [item["crop_id"] for item in response.json()["items"]] == [oldest, newest]
    assert len(calls) == 2
    missing = client.post(
        "/api/face/diagnostics/crops", json={"crop_ids": [str(uuid.uuid4())]}
    )
    assert missing.status_code == 200
    assert missing.json()["items"] == []
    assert len(calls) == 2


@pytest.mark.parametrize("crop_ids", [[], ["not-a-uuid"], [str(uuid.uuid4())] * 101])
def test_face_diagnostic_batches_are_validated_before_model_work(client, crop_ids):
    response = client.post("/api/face/diagnostics/crops", json={"crop_ids": crop_ids})
    assert response.status_code == 422


@pytest.mark.parametrize("path", ["/api/does-not-exist", "/v1/does-not-exist"])
def test_unknown_api_routes_return_json_not_the_spa(client, path):
    response = client.get(path)
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"] == "API endpoint not found"
