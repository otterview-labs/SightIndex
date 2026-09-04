"""Trajectory mode=reid: sources, thresholds, and observable degradation."""

import importlib
import sys
import uuid
from datetime import datetime

from fastapi.testclient import TestClient


def load_app(monkeypatch, tmp_path, name: str, **env: str):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / f'{name}.db'}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PERSON_DETECTOR", "whole_frame")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for module_name in list(sys.modules):
        if module_name == "main" or module_name.startswith("app."):
            sys.modules.pop(module_name)
    return importlib.import_module("main")


def _seed_person_with_crops(db_module, tmp_path, *, with_files=True):
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person

    crops_dir = tmp_path / "data" / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    with db_module() as db:
        person = Person(name="张三")
        db.add(person)
        db.flush()
        captured_at = datetime(2026, 8, 23, 10, 0, 0)
        image = Image(
            image_url="/data/frames/t.jpg",
            source_type="stream_frame",
            captured_at=captured_at,
        )
        db.add(image)
        db.flush()
        ids = {}
        for name in ("seed", "hit", "weak"):
            if with_files:
                (crops_dir / (name + ".jpg")).write_bytes(b"img")
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/" + name + ".jpg",
                bbox={"label": "person"},
                camera_id=uuid.uuid4(),
                person_id=person.id if name == "seed" else None,
                captured_at=captured_at,
            )
            db.add(crop)
            db.flush()
            ids[name] = crop.id
        db.commit()
        return person.id, ids


def test_trajectory_mode_reid_labels_and_thresholds(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "test-traj-reid",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
        REID_GALLERY_TOP_K="1",
        REID_MIN_SCORE="0.5",
    )

    from app.db.session import SessionLocal
    from app.services.reid import ReidEmbeddingService
    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    with TestClient(main.create_app()) as client:
        person_id, ids = _seed_person_with_crops(SessionLocal, tmp_path)

        monkeypatch.setattr(
            ReidEmbeddingService,
            "embed_images",
            lambda self, paths: [[0.1] * self.dim for _ in paths],
        )
        # `hit` clears REID_MIN_SCORE; `weak` passes the per-vote bar but not aggregation.
        monkeypatch.setattr(
            MilvusVectorIndex,
            "search_vector",
            lambda self, ot, vec, top_k: [
                VectorSearchHit(object_id=ids["hit"], score=0.92),
                VectorSearchHit(object_id=ids["weak"], score=0.55),
            ],
        )
        monkeypatch.setattr(
            "app.services.reid_index.ReidIndexService.gallery_matches",
            lambda self, paths, top_k=None, deadline=None, warnings=None: {
                ids["hit"]: 0.92,
                ids["weak"]: 0.31,
            },
        )

        response = client.get(f"/api/persons/{person_id}/trajectory?mode=reid")

    assert response.status_code == 200
    payload = response.json()
    assert any("结果可能不完整" in warning for warning in payload["warnings"])
    # Post-aggregation threshold: weak aggregated to 0.31 < 0.5 and must not appear.
    assert [item["crop_id"] for item in payload["items"]] == [str(ids["hit"])]
    assert payload["items"][0]["match_source"] == "reid"
    assert payload["items"][0]["result_type"] == "reid_match"
    assert payload["items"][0]["vector_score"] == 0.92


def test_trajectory_mode_reid_reports_failure_instead_of_empty(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "test-traj-reid-fail",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )

    from app.db.session import SessionLocal
    from app.services.reid import ReidRuntimeError

    with TestClient(main.create_app()) as client:
        person_id, _ = _seed_person_with_crops(SessionLocal, tmp_path)

        def boom(self, paths, top_k=None, deadline=None, warnings=None):
            raise ReidRuntimeError("connection refused")

        monkeypatch.setattr(
            "app.services.reid_index.ReidIndexService.gallery_matches", boom
        )
        response = client.get(f"/api/persons/{person_id}/trajectory?mode=reid")

    payload = response.json()
    assert payload["items"] == []
    # A dead service must be distinguishable from a genuine "no matches".
    assert any("ReID 检索失败" in warning for warning in payload["warnings"])


def test_trajectory_mode_reid_warns_without_body_crop_seeds(monkeypatch, tmp_path):
    """Face-enrollment portraits are not silently substituted as whole-body gallery seeds."""

    main = load_app(
        monkeypatch,
        tmp_path,
        "test-traj-reid-noseed",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )

    from app.db.session import SessionLocal
    from app.models.persons import Person
    from app.services.vector_index import MilvusVectorIndex

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)

    with TestClient(main.create_app()) as client:
        with SessionLocal() as db:
            person = Person(name="李四", avatar_url="/data/uploads/portrait.jpg")
            db.add(person)
            db.commit()
            person_id = person.id
        response = client.get(f"/api/persons/{person_id}/trajectory?mode=reid")

    payload = response.json()
    assert payload["items"] == []
    assert any("人体裁剪" in warning for warning in payload["warnings"])


def test_trajectory_excludes_crops_assigned_to_other_people(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "test-traj-reid-exclude",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )

    from app.db.session import SessionLocal
    from app.models.media import PersonCrop
    from app.models.persons import Person

    with TestClient(main.create_app()) as client:
        person_id, ids = _seed_person_with_crops(SessionLocal, tmp_path)
        with SessionLocal() as db:
            other = Person(name="别人")
            db.add(other)
            db.flush()
            hit = db.get(PersonCrop, ids["hit"])
            hit.person_id = other.id
            db.add(hit)
            db.commit()

        monkeypatch.setattr(
            "app.services.reid_index.ReidIndexService.gallery_matches",
            lambda self, paths, top_k=None, deadline=None, warnings=None: {
                ids["hit"]: 0.95
            },
        )
        response = client.get(f"/api/persons/{person_id}/trajectory?mode=reid")

    payload = response.json()
    # A crop confirmed as someone else can score arbitrarily high and still must not appear.
    assert payload["items"] == []


def test_trajectory_reid_applies_requested_min_similarity(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "test-traj-reid-min-score",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )
    from app.db.session import SessionLocal

    with TestClient(main.create_app()) as client:
        person_id, ids = _seed_person_with_crops(SessionLocal, tmp_path)
        monkeypatch.setattr(
            "app.services.reid_index.ReidIndexService.gallery_matches",
            lambda self, paths, top_k=None, deadline=None, warnings=None: {
                ids["hit"]: 0.7
            },
        )
        response = client.get(
            f"/api/persons/{person_id}/trajectory?mode=reid&min_similarity=0.8"
        )

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_trajectory_reid_excludes_non_camera_uploads(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "test-traj-reid-camera-only",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )
    from app.db.session import SessionLocal
    from app.models.media import PersonCrop

    with TestClient(main.create_app()) as client:
        person_id, ids = _seed_person_with_crops(SessionLocal, tmp_path)
        with SessionLocal() as db:
            hit = db.get(PersonCrop, ids["hit"])
            assert hit is not None
            hit.camera_id = None
            db.commit()
        monkeypatch.setattr(
            "app.services.reid_index.ReidIndexService.gallery_matches",
            lambda self, paths, top_k=None, deadline=None, warnings=None: {
                ids["hit"]: 0.95
            },
        )
        response = client.get(f"/api/persons/{person_id}/trajectory?mode=reid")

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_trajectory_reid_excludes_crop_without_real_capture_time(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "test-traj-reid-capture-time",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop

    with TestClient(main.create_app()) as client:
        person_id, ids = _seed_person_with_crops(SessionLocal, tmp_path)
        with SessionLocal() as db:
            hit = db.get(PersonCrop, ids["hit"])
            assert hit is not None
            image = db.get(Image, hit.image_id)
            assert image is not None
            hit.captured_at = None
            image.captured_at = None
            db.commit()
        monkeypatch.setattr(
            "app.services.reid_index.ReidIndexService.gallery_matches",
            lambda self, paths, top_k=None, deadline=None, warnings=None: {
                ids["hit"]: 0.95
            },
        )
        response = client.get(f"/api/persons/{person_id}/trajectory?mode=reid")

    payload = response.json()
    assert response.status_code == 200
    assert payload["items"] == []
    assert any("缺少真实采集时间" in warning for warning in payload["warnings"])


def test_trajectory_rejects_reversed_time_range(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-traj-time-range")
    from app.db.session import SessionLocal

    with TestClient(main.create_app()) as client:
        person_id, _ = _seed_person_with_crops(SessionLocal, tmp_path)
        response = client.get(
            f"/api/persons/{person_id}/trajectory"
            "?start_time=2026-08-24T00:00:00Z&end_time=2026-08-23T00:00:00Z"
        )

    assert response.status_code == 422
    assert "start_time" in response.json()["detail"]


def test_trajectory_naive_database_time_is_interpreted_in_local_timezone(
    monkeypatch, tmp_path
):
    main = load_app(
        monkeypatch,
        tmp_path,
        "test-traj-timezone",
        LOCAL_TIMEZONE="Asia/Shanghai",
    )
    from datetime import UTC, datetime

    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.services.observation_index import ObservationIndexService

    # SQLite round-trips DateTime(timezone=True) as naive wall-clock values. Here 10:00 is
    # Shanghai local time (02:00 UTC), so an API boundary of 02:30 UTC must include it.
    captured_at = datetime(2026, 8, 23, 10, 0, 0)
    end_time = datetime(2026, 8, 23, 2, 30, 0, tzinfo=UTC)
    with TestClient(main.create_app()) as client:
        with SessionLocal() as db:
            person = Person(name="时区回归")
            db.add(person)
            db.flush()
            image = Image(
                image_url="/data/frames/timezone.jpg",
                source_type="stream_frame",
                captured_at=captured_at,
            )
            db.add(image)
            db.flush()
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/timezone.jpg",
                bbox={"label": "person"},
                person_id=person.id,
                captured_at=captured_at,
            )
            db.add(crop)
            db.flush()
            ObservationIndexService(db, main.get_settings()).upsert_crop(crop)
            db.commit()
            person_id = person.id
            crop_id = crop.id

        response = client.get(
            f"/api/persons/{person_id}/trajectory",
            params={"mode": "face", "end_time": end_time.isoformat()},
        )

    assert response.status_code == 200
    payload = response.json()
    assert [item["crop_id"] for item in payload["items"]] == [str(crop_id)]


def test_schema_replaces_synthetic_observation_created_at_with_recognition_time(
    monkeypatch, tmp_path
):
    main = load_app(
        monkeypatch,
        tmp_path,
        "test-observation-synthetic-time-migration",
        LOCAL_TIMEZONE="Asia/Shanghai",
    )
    from app.db.session import SessionLocal
    from app.models.events import RecognitionEvent
    from app.models.media import Image, PersonCrop, PersonObservationIndex

    main.init_db()
    recognized_at = datetime(2026, 8, 23, 19, 5, 0)
    with SessionLocal() as db:
        image = Image(image_url="/data/uploads/no-time.jpg", source_type="upload")
        db.add(image)
        db.flush()
        crop = PersonCrop(
            image_id=image.id,
            crop_url="/data/crops/no-time.jpg",
            bbox={},
        )
        db.add(crop)
        db.flush()
        db.add(
            RecognitionEvent(
                image_id=image.id,
                crop_id=crop.id,
                result_type="unknown",
                recognized_at=recognized_at,
            )
        )
        # This is the legacy fallback that mixed SQLite UTC CURRENT_TIMESTAMP into captured_at.
        observation = PersonObservationIndex(
            crop_id=crop.id,
            image_id=image.id,
            captured_at=crop.created_at,
        )
        db.add(observation)
        db.commit()
        observation_id = observation.id

    main.init_db()

    with SessionLocal() as db:
        migrated = db.get(PersonObservationIndex, observation_id)
        assert migrated is not None
        assert migrated.captured_at == recognized_at


def test_trajectory_face_mode_carries_bbox_and_null_capture_time(monkeypatch, tmp_path):
    """The regression the long-failing smoke test was pointing at, pinned directly."""

    main = load_app(monkeypatch, tmp_path, "test-traj-face-bbox")

    from app.db.session import SessionLocal
    from app.models.events import RecognitionEvent
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.services.observation_index import ObservationIndexService
    from app.services.time_utils import local_now

    with TestClient(main.create_app()) as client:
        with SessionLocal() as db:
            person = Person(name="王五")
            db.add(person)
            db.flush()
            image = Image(image_url="/data/frames/f.jpg", source_type="upload")
            db.add(image)
            db.flush()
            # captured_at deliberately absent: uploads have no camera timestamp.
            crop = PersonCrop(image_id=image.id, crop_url="/data/crops/f.jpg", bbox={})
            db.add(crop)
            db.flush()
            event = RecognitionEvent(
                person_id=person.id,
                image_id=image.id,
                crop_id=crop.id,
                result_type="known",
                similarity=0.99,
                face_bbox={"x": 5.0, "y": 6.0, "width": 40.0, "height": 44.0},
                recognized_at=local_now(main.get_settings()),
            )
            db.add(event)
            crop.person_id = person.id
            ObservationIndexService(db, main.get_settings()).upsert_crop(crop)
            db.commit()
            person_id = person.id

        response = client.get(f"/api/persons/{person_id}/trajectory?min_similarity=0.1")

    payload = response.json()
    assert len(payload["items"]) == 1
    item = payload["items"][0]
    assert item["match_source"] == "face"
    assert item["similarity"] >= 0.99
    assert item["face_bbox"]["width"] == 40.0
