"""Naming a person from a body crop, with no face involved."""
import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from test_reid import load_app


@pytest.fixture
def client_and_ids(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-crop-labels")

    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop

    ids: dict[str, uuid.UUID] = {}
    with TestClient(main.create_app()) as client:
        with SessionLocal() as db:
            image = Image(image_url="/data/frames/f.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            for name in ("one", "two"):
                crop = PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/{name}.jpg",
                    bbox={"label": "person"},
                )
                db.add(crop)
                db.commit()
                db.refresh(crop)
                ids[name] = crop.id
        person = client.post("/api/persons", json={"name": "张三"}).json()
        ids["person"] = person["id"]
        yield client, ids


def test_labelling_a_crop_needs_no_face(client_and_ids):
    client, ids = client_and_ids

    response = client.post(f"/api/persons/{ids['person']}/crops/{ids['one']}")

    assert response.status_code == 200
    assert response.json()["person_id"] == ids["person"]


def test_the_label_reaches_the_observation_table(client_and_ids):
    """Without this the name exists but never shows up anywhere the operator looks."""

    client, ids = client_and_ids
    client.post(f"/api/persons/{ids['person']}/crops/{ids['one']}")

    rows = client.get("/api/search/observations?limit=50").json()["items"]
    row = next(r for r in rows if r["crop_id"] == str(ids["one"]))

    assert row["person_name"] == "张三"
    # A manual label is an assertion, not a recognition; it must not look like a face match.
    assert row["recognition_result_type"] is None
    assert row["face_similarity"] is None


def test_a_label_seeds_the_reid_gallery(client_and_ids):
    client, ids = client_and_ids
    client.post(f"/api/persons/{ids['person']}/crops/{ids['one']}")
    client.post(f"/api/persons/{ids['person']}/crops/{ids['two']}")

    gallery = client.get(f"/api/persons/{ids['person']}/crops").json()

    assert {item["id"] for item in gallery} == {str(ids["one"]), str(ids["two"])}


def test_the_first_label_becomes_the_avatar(client_and_ids):
    client, ids = client_and_ids
    client.post(f"/api/persons/{ids['person']}/crops/{ids['one']}")

    person = client.get(f"/api/persons/{ids['person']}").json()

    assert person["avatar_url"] == "/data/crops/one.jpg"


def test_relabelling_someone_elses_crop_is_refused(client_and_ids):
    """Silently stealing a crop would corrupt both galleries at once."""

    client, ids = client_and_ids
    client.post(f"/api/persons/{ids['person']}/crops/{ids['one']}")
    other = client.post("/api/persons", json={"name": "李四"}).json()

    response = client.post(f"/api/persons/{other['id']}/crops/{ids['one']}")

    assert response.status_code == 409
    assert client.get(f"/api/persons/{other['id']}/crops").json() == []


def test_a_label_can_be_taken_back(client_and_ids):
    client, ids = client_and_ids
    client.post(f"/api/persons/{ids['person']}/crops/{ids['one']}")

    response = client.delete(f"/api/persons/{ids['person']}/crops/{ids['one']}")

    assert response.status_code == 200
    assert response.json()["person_id"] is None
    assert client.get(f"/api/persons/{ids['person']}/crops").json() == []
    rows = client.get("/api/search/observations?limit=50").json()["items"]
    row = next(r for r in rows if r["crop_id"] == str(ids["one"]))
    assert row["person_name"] is None


def test_unlabelling_a_crop_that_is_not_theirs_is_a_404(client_and_ids):
    client, ids = client_and_ids

    response = client.delete(f"/api/persons/{ids['person']}/crops/{ids['two']}")

    assert response.status_code == 404


def test_a_person_starts_out_not_vip(client_and_ids):
    client, ids = client_and_ids

    person = client.get(f"/api/persons/{ids['person']}").json()

    assert person["is_vip"] is False


def test_vip_is_a_manual_toggle(client_and_ids):
    client, ids = client_and_ids

    response = client.patch(f"/api/persons/{ids['person']}", json={"is_vip": True})

    assert response.status_code == 200
    assert response.json()["is_vip"] is True
    assert client.get(f"/api/persons/{ids['person']}").json()["is_vip"] is True

    response = client.patch(f"/api/persons/{ids['person']}", json={"is_vip": False})
    assert response.json()["is_vip"] is False


def test_vip_status_reaches_the_observation_table(client_and_ids):
    """Denormalized the same way person_name is, so a search row never needs a join for it."""

    client, ids = client_and_ids
    client.patch(f"/api/persons/{ids['person']}", json={"is_vip": True})
    client.post(f"/api/persons/{ids['person']}/crops/{ids['one']}")

    rows = client.get("/api/search/observations?limit=50").json()["items"]
    row = next(r for r in rows if r["crop_id"] == str(ids["one"]))

    assert row["person_is_vip"] is True


def test_toggling_vip_updates_already_indexed_observation_rows(client_and_ids):
    """The flag can change after a crop was labelled; old rows must not stay stale forever."""

    client, ids = client_and_ids
    client.post(f"/api/persons/{ids['person']}/crops/{ids['one']}")
    rows = client.get("/api/search/observations?limit=50").json()["items"]
    row = next(r for r in rows if r["crop_id"] == str(ids["one"]))
    assert row["person_is_vip"] is False

    client.patch(f"/api/persons/{ids['person']}", json={"is_vip": True})

    rows = client.get("/api/search/observations?limit=50").json()["items"]
    row = next(r for r in rows if r["crop_id"] == str(ids["one"]))
    assert row["person_is_vip"] is True


def test_visit_stats_for_a_never_seen_person_are_all_empty(client_and_ids):
    """Being named is not being seen: no observation rows exist yet for this person."""

    client, ids = client_and_ids

    stats = client.get(f"/api/persons/{ids['person']}/visit-stats").json()

    assert stats["total_appearances"] == 0
    assert stats["first_seen"] is None
    assert stats["last_seen"] is None
    assert stats["visit_days_in_window"] == 0
    assert stats["is_repeat_visitor"] is False


def test_repeat_visitor_status_follows_distinct_days_not_raw_appearances(monkeypatch, tmp_path):
    """Multiple crops on one calendar day do not count as separate visit days."""

    main = load_app(monkeypatch, tmp_path, "test-visit-stats", REPEAT_VISITOR_MIN_DAYS="3")

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.observation_index import ObservationIndexService
    from app.services.time_utils import local_now

    # Anchored to the test's own current time (at noon, far from any midnight boundary) rather
    # than a fixed calendar date, so this does not go stale as "now" moves past the trailing
    # window the endpoint measures against.
    today = local_now(get_settings()).replace(
        hour=12, minute=0, second=0, microsecond=0, tzinfo=None
    )

    with TestClient(main.create_app()) as client:
        person = client.post("/api/persons", json={"name": "老王"}).json()
        with SessionLocal() as db:
            image = Image(image_url="/data/frames/f.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            image_id = image.id
            settings = get_settings()
            for hour_offset in (0, 1, 2):
                crop = PersonCrop(
                    image_id=image_id,
                    crop_url=f"/data/crops/same-day-{hour_offset}.jpg",
                    bbox={"label": "person"},
                    person_id=uuid.UUID(person["id"]),
                    captured_at=today + timedelta(hours=hour_offset),
                )
                db.add(crop)
                db.commit()
                db.refresh(crop)
                ObservationIndexService(db, settings).upsert_crop(crop)
                db.commit()

        stats = client.get(f"/api/persons/{person['id']}/visit-stats").json()
        assert stats["total_appearances"] == 3
        assert stats["visit_days_in_window"] == 1
        assert stats["is_repeat_visitor"] is False

        with SessionLocal() as db:
            for day_offset in (1, 2):
                crop = PersonCrop(
                    image_id=image_id,
                    crop_url=f"/data/crops/day-{day_offset}.jpg",
                    bbox={"label": "person"},
                    person_id=uuid.UUID(person["id"]),
                    captured_at=today - timedelta(days=day_offset),
                )
                db.add(crop)
                db.commit()
                db.refresh(crop)
                ObservationIndexService(db, settings).upsert_crop(crop)
                db.commit()

        stats = client.get(f"/api/persons/{person['id']}/visit-stats").json()
        assert stats["total_appearances"] == 5
        assert stats["visit_days_in_window"] == 3
        assert stats["is_repeat_visitor"] is True


def test_a_reid_trajectory_point_names_its_camera(monkeypatch, tmp_path):
    """A point carrying only a uuid renders as `camera 2c2a45b3`, which locates nothing."""

    main = load_app(monkeypatch, tmp_path, "test-traj-names")

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop, VideoStream
    from app.models.persons import Person
    from app.services.persons import PersonService

    camera_id = uuid.uuid4()
    with TestClient(main.create_app()):
        with SessionLocal() as db:
            db.add(
                VideoStream(
                    name="项目部门口",
                    stream_url="rtsp://example/1",
                    camera_id=camera_id,
                    location_name="研发中心 3F 项目部",
                )
            )
            person = Person(name="张三")
            image = Image(image_url="/data/frames/f.jpg", source_type="stream_frame")
            db.add_all([person, image])
            db.commit()
            db.refresh(person)
            db.refresh(image)
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/one.jpg",
                bbox={"label": "person"},
                camera_id=camera_id,
                captured_at=datetime(2026, 8, 24, 13, 50, 0),
            )
            db.add(crop)
            db.commit()
            db.refresh(crop)

            point = PersonService(db, get_settings())._vector_trajectory_point(
                crop=crop,
                person=person,
                event=None,
                counting_event=None,
                image=image,
                vector_score=0.81,
                source="reid",
            )

    assert point.camera_name == "项目部门口"
    assert point.location_name == "研发中心 3F 项目部"
    assert point.vector_score == 0.81
