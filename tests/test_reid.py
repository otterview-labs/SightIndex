import importlib
import sys
import uuid
from datetime import datetime, timedelta

import pytest
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


def test_raw_vector_paths_need_only_milvus(monkeypatch, tmp_path):
    """ReID brings its own vectors, so it must not depend on the text/visual embedding config."""

    load_app(monkeypatch, tmp_path, "test-reid-raw-enable", MILVUS_ENABLED="true")

    from app.config.settings import get_settings
    from app.services.vector_index import MilvusVectorIndex

    index = MilvusVectorIndex(get_settings())
    # No embedding provider configured: the embed-for-you paths stay off...
    assert index.is_enabled() is False
    # ...but the raw-vector paths only need Milvus itself.
    assert index.is_available() is True


def test_status_separates_configured_from_ready(monkeypatch, tmp_path):
    """REID_* being set is not readiness; ready requires the service to answer its health check."""

    main = load_app(
        monkeypatch,
        tmp_path,
        "test-reid-ready",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )

    from app.services.reid import ReidEmbeddingService
    from app.services.vector_index import MilvusVectorIndex

    # ready requires BOTH probes; hold Milvus green here to isolate the service leg.
    monkeypatch.setattr(
        MilvusVectorIndex, "probe", lambda self, timeout_seconds=2.0: (True, None)
    )
    monkeypatch.setattr(
        ReidEmbeddingService,
        "probe",
        lambda self, timeout_seconds=3.0: (False, "connection refused"),
    )
    with TestClient(main.create_app()) as client:
        payload = client.get("/api/reid/status").json()
    assert payload["enabled"] is True
    assert payload["ready"] is False
    assert payload["reid_service_ok"] is False
    assert payload["milvus_configured"] is True
    assert payload["milvus_ok"] is True
    assert "connection refused" in payload["last_error"]

    monkeypatch.setattr(
        ReidEmbeddingService, "probe", lambda self, timeout_seconds=3.0: (True, None)
    )
    with TestClient(main.create_app()) as client:
        payload = client.get("/api/reid/status").json()
    assert payload["ready"] is True
    assert payload["last_error"] is None
    assert payload["checkpoint_revision"].startswith("sha256:")
    assert payload["milvus_namespace"].startswith("endpoint-")
    assert payload["index_fingerprint"].startswith("sapiensid_wb12m|sha256:")


def test_reid_status_reports_disabled_without_configuration(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-reid-off")
    with TestClient(main.create_app()) as client:
        payload = client.get("/api/reid/status").json()
    assert payload["enabled"] is False
    assert payload["embedding_dim"] == 4096
    assert payload["indexed_crops"] == 0


def test_reid_endpoints_explain_missing_configuration(monkeypatch, tmp_path):
    """A 503 naming the settings beats a stack trace when ReID is simply not turned on."""

    main = load_app(monkeypatch, tmp_path, "test-reid-503")
    with TestClient(main.create_app()) as client:
        response = client.post(
            "/api/reid/search",
            files={"file": ("query.jpg", b"not-a-real-image", "image/jpeg")},
        )
    assert response.status_code == 503
    assert "REID_ENABLED" in response.json()["detail"]


def test_reid_attribute_filter_tolerates_one_conflict_and_rejects_two(
    monkeypatch, tmp_path
):
    main = load_app(monkeypatch, tmp_path, "test-reid-attribute-filter")
    from app.api.reid import _filter_by_attributes, _reject_attribute_conflicts
    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.schemas.reid import ReidMatchItem

    query = {
        "clothing": {
            "upper_color": "black",
            "lower_color": "black",
            "upper_color_confidence": 0.95,
            "lower_color_confidence": 0.95,
        }
    }
    with TestClient(main.create_app()):
        with SessionLocal() as db:
            image = Image(image_url="/data/frame.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            one_conflict = PersonCrop(
                image_id=image.id,
                crop_url="/data/one-conflict.jpg",
                bbox={"label": "person"},
                attributes={
                    "clothing": {
                        "upper_color": "white",
                        "lower_color": "black",
                        "upper_color_confidence": 0.92,
                        "lower_color_confidence": 0.91,
                    }
                },
            )
            two_conflicts = PersonCrop(
                image_id=image.id,
                crop_url="/data/two-conflicts.jpg",
                bbox={"label": "person"},
                attributes={
                    "clothing": {
                        "upper_color": "white",
                        "lower_color": "white",
                        "upper_color_confidence": 0.94,
                        "lower_color_confidence": 0.93,
                    }
                },
            )
            unknown = PersonCrop(
                image_id=image.id,
                crop_url="/data/unknown.jpg",
                bbox={"label": "person"},
                attributes={"source": "cv_tone", "clothing": {"upper_color": "dark"}},
            )
            db.add_all([one_conflict, two_conflicts, unknown])
            db.commit()
            db.refresh(one_conflict)
            db.refresh(two_conflicts)
            db.refresh(unknown)

            items, bonuses = _filter_by_attributes(
                db,
                get_settings(),
                [
                    ReidMatchItem(crop_id=one_conflict.id, score=0.9),
                    ReidMatchItem(crop_id=two_conflicts.id, score=0.85),
                    ReidMatchItem(crop_id=unknown.id, score=0.8),
                ],
                query,
            )

            assert get_settings().reid_attribute_hard_conflicts == 2
            now = datetime.now().astimezone()
            for item in items:
                item.captured_at = now
            filtered = _reject_attribute_conflicts(
                items,
                get_settings(),
                query_captured_at=now,
            )

    assert [item.crop_id for item in items] == [
        one_conflict.id,
        two_conflicts.id,
        unknown.id,
    ]
    assert [item.crop_id for item in filtered] == [one_conflict.id, unknown.id]
    assert items[0].attribute_agreement == pytest.approx(0.91 / 1.83)
    assert items[0].attribute_conflicts == ["upper_color"]
    assert items[0].attribute_evidence_weight == 1.83
    assert items[0].attribute_conflict_weight == 0.92
    assert items[2].attribute_agreement is None
    assert bonuses[one_conflict.id] < 0
    assert bonuses[two_conflicts.id] < bonuses[one_conflict.id]

    items[1].face_match = True
    assert _reject_attribute_conflicts(
        items,
        get_settings(),
        query_captured_at=now,
    ) == items

    items[1].face_match = None
    items[1].captured_at = now - timedelta(days=1)
    assert _reject_attribute_conflicts(
        items,
        get_settings(),
        query_captured_at=now,
    ) == items


def test_reid_search_returns_matches_enriched_from_the_observation_index(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "test-reid-search",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.observation_index import ObservationIndexService
    from app.services.reid import ReidEmbeddingService
    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    crop_ids: dict[str, uuid.UUID] = {}

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)
    monkeypatch.setattr(
        ReidEmbeddingService, "embed_image", lambda self, path: [0.1] * self.dim
    )

    def fake_search(self, object_type, vector, top_k):
        assert object_type == "reid_person_crop"
        assert len(vector) == 4096
        return [
            VectorSearchHit(object_id=crop_ids["hit"], score=0.91),
            VectorSearchHit(object_id=crop_ids["weak"], score=0.20),
        ]

    monkeypatch.setattr(MilvusVectorIndex, "search_vector", fake_search)

    with TestClient(main.create_app()) as client:
        with SessionLocal() as db:
            image = Image(image_url="/data/frames/f.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            for name in ("hit", "weak"):
                crop = PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/{name}.jpg",
                    bbox={"label": "person"},
                )
                db.add(crop)
                db.commit()
                db.refresh(crop)
                crop_ids[name] = crop.id
                ObservationIndexService(db, get_settings()).upsert_crop(crop)
            db.commit()

        response = client.post(
            "/api/reid/search",
            files={"file": ("query.jpg", b"bytes", "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    # REID_MIN_SCORE defaults to 0.5, so the 0.20 hit is filtered out.
    assert [item["crop_id"] for item in payload["items"]] == [str(crop_ids["hit"])]
    assert payload["items"][0]["score"] == pytest.approx(0.91)
    assert payload["items"][0]["crop_url"] == "/data/crops/hit.jpg"
    assert payload["items"][0]["fusion_score"] == pytest.approx(0.91)
    assert payload["items"][0]["evidence_level"] == "similar"
    assert "人体达到候选范围" in payload["items"][0]["decision_reason"]
    assert payload["model"] == "sapiensid_wb12m"


def test_gallery_scoring_averages_the_best_votes(monkeypatch, tmp_path):
    """Consistent recognition across the gallery must outrank one lucky frame."""

    load_app(
        monkeypatch,
        tmp_path,
        "test-reid-gallery",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
        REID_GALLERY_TOP_K="2",
    )

    from pathlib import Path

    from app.config.settings import get_settings
    from app.services.reid import ReidEmbeddingService
    from app.services.reid_index import ReidIndexService
    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    steady = uuid.uuid4()
    lucky = uuid.uuid4()

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)
    monkeypatch.setattr(
        ReidEmbeddingService,
        "embed_images",
        lambda self, paths: [[0.1] * self.dim for _ in paths],
    )

    # Three gallery images vote. `steady` is recognised by all three; `lucky` is matched once.
    votes = iter(
        [
            [
                VectorSearchHit(object_id=steady, score=0.80),
                VectorSearchHit(object_id=lucky, score=0.99),
            ],
            [VectorSearchHit(object_id=steady, score=0.90)],
            [VectorSearchHit(object_id=steady, score=0.70)],
        ]
    )
    monkeypatch.setattr(MilvusVectorIndex, "search_vector", lambda self, *a: next(votes))

    service = ReidIndexService(None, get_settings())
    scores = service.gallery_matches([Path("a.jpg"), Path("b.jpg"), Path("c.jpg")])

    # steady was recognised three times: its two best (0.90, 0.80) over a denominator of 2.
    assert scores[steady] == pytest.approx(0.85)
    # lucky matched once at 0.99, but the missing second vote counts as zero.
    assert scores[lucky] == pytest.approx(0.495)
    # Which is the whole point: consistent recognition outranks one lucky frame.
    assert max(scores, key=lambda key: scores[key]) == steady


def test_gallery_deadline_stops_before_another_milvus_query(monkeypatch, tmp_path):
    load_app(
        monkeypatch,
        tmp_path,
        "test-reid-gallery-deadline",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )

    from pathlib import Path

    from app.config.settings import get_settings
    from app.services import reid_index
    from app.services.reid import ReidEmbeddingService, ReidRuntimeError
    from app.services.reid_index import ReidIndexService
    from app.services.vector_index import MilvusVectorIndex

    monkeypatch.setattr(
        ReidEmbeddingService,
        "embed_images",
        lambda self, paths, deadline=None: [[0.1] * self.dim for _ in paths],
    )
    searches = 0

    def search(self, object_type, vector, top_k):
        nonlocal searches
        searches += 1
        return []

    monkeypatch.setattr(MilvusVectorIndex, "search_vector", search)
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(reid_index, "monotonic", lambda: next(clock))

    with pytest.raises(ReidRuntimeError, match="deadline exceeded"):
        ReidIndexService(None, get_settings()).gallery_matches(
            [Path("a.jpg"), Path("b.jpg")],
            deadline=1.0,
        )
    assert searches == 1


def test_gallery_bad_input_isolates_one_corrupt_seed(monkeypatch, tmp_path):
    load_app(
        monkeypatch,
        tmp_path,
        "test-reid-gallery-bad-seed",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
        REID_GALLERY_TOP_K="1",
    )
    from pathlib import Path

    from app.config.settings import get_settings
    from app.services.reid import ReidEmbeddingService, ReidRuntimeError
    from app.services.reid_index import ReidIndexService
    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    monkeypatch.setattr(
        ReidEmbeddingService,
        "embed_images",
        lambda self, paths: (_ for _ in ()).throw(
            ReidRuntimeError("bad batch", status_code=422)
        ),
    )

    def embed_one(self, path, deadline=None):
        if path.name == "bad.jpg":
            raise ReidRuntimeError("bad image", status_code=422)
        return [0.1] * self.dim

    monkeypatch.setattr(ReidEmbeddingService, "embed_image", embed_one)
    hit_id = uuid.uuid4()
    monkeypatch.setattr(
        MilvusVectorIndex,
        "search_vector",
        lambda self, object_type, vector, top_k: [
            VectorSearchHit(object_id=hit_id, score=0.9)
        ],
    )
    warnings: list[str] = []

    scores = ReidIndexService(None, get_settings()).gallery_matches(
        [Path("bad.jpg"), Path("good.jpg")],
        warnings=warnings,
    )

    assert scores == {hit_id: pytest.approx(0.9)}
    assert any("跳过 1" in warning for warning in warnings)


def test_similar_to_crop_drops_the_query_before_collapsing(monkeypatch, tmp_path):
    """The query scores 1.0 on itself; collapsing first would hand it its own visit back."""

    main = load_app(
        monkeypatch,
        tmp_path,
        "test-reid-collapse",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.observation_index import ObservationIndexService
    from app.services.reid import ReidEmbeddingService
    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    query_camera_id = uuid.uuid4()
    candidate_camera_id = uuid.uuid4()
    captured = datetime(2026, 8, 24, 9, 40, 0)
    crop_ids: dict[str, uuid.UUID] = {}
    scores = {"query": 1.0, "home": 0.94, "neighbour": 0.88, "later": 0.71}

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)
    monkeypatch.setattr(
        ReidEmbeddingService, "embed_image", lambda self, path: [0.1] * self.dim
    )
    monkeypatch.setattr(
        MilvusVectorIndex,
        "search_vector",
        lambda self, object_type, vector, top_k: [
            VectorSearchHit(object_id=crop_ids[name], score=score)
            for name, score in scores.items()
        ],
    )

    with TestClient(main.create_app()) as client:
        with SessionLocal() as db:
            image = Image(image_url="/data/frames/f.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            # search_by_crop reads the crop off disk, so the files have to be there.
            crop_dir = get_settings().data_dir / "crops"
            crop_dir.mkdir(parents=True, exist_ok=True)
            # The query itself is removed, while both its home-camera match and cross-camera
            # visits remain visible and are grouped by camera in the frontend.
            for name, offset in (
                ("query", 0),
                ("home", 120),
                ("neighbour", 2),
                ("later", 600),
            ):
                (crop_dir / f"{name}.jpg").write_bytes(b"bytes")
                crop = PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/{name}.jpg",
                    bbox={"label": "person"},
                    camera_id=(
                        query_camera_id
                        if name in {"query", "home"}
                        else candidate_camera_id
                    ),
                    captured_at=captured + timedelta(seconds=offset),
                )
                db.add(crop)
                db.commit()
                db.refresh(crop)
                crop_ids[name] = crop.id
                ObservationIndexService(db, get_settings()).upsert_crop(crop)
            db.commit()

        response = client.post(f"/api/reid/crops/{crop_ids['query']}/similar")

    assert response.status_code == 200
    payload = response.json()
    assert payload["collapse_window_seconds"] == 60.0
    # The home-camera visit is visible too, while the query itself is never returned.
    assert [item["crop_id"] for item in payload["items"]] == [
        str(crop_ids["home"]),
        str(crop_ids["neighbour"]),
        str(crop_ids["later"]),
    ]
    assert [item["frame_count"] for item in payload["items"]] == [1, 1, 1]


def test_matches_are_placed_even_before_the_observation_row_lands(monkeypatch, tmp_path):
    """The observation row trails ingest by seconds; the freshest crops must still collapse."""

    main = load_app(
        monkeypatch,
        tmp_path,
        "test-reid-lag",
        REID_ENABLED="true",
        REID_SERVICE_URL="http://reid.local",
        MILVUS_ENABLED="true",
    )

    from app.db.session import SessionLocal
    from app.models.media import (
        Image,
        PersonCrop,
        PersonObservationIndex,
        VideoStream,
    )
    from app.services.reid import ReidEmbeddingService
    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    camera_id = uuid.uuid4()
    captured = datetime(2026, 8, 24, 11, 4, 0)
    crop_ids: list[uuid.UUID] = []

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)
    monkeypatch.setattr(
        ReidEmbeddingService, "embed_image", lambda self, path: [0.1] * self.dim
    )
    monkeypatch.setattr(
        MilvusVectorIndex,
        "search_vector",
        lambda self, object_type, vector, top_k: [
            VectorSearchHit(object_id=crop_id, score=0.9 - index * 0.01)
            for index, crop_id in enumerate(crop_ids)
        ],
    )

    with TestClient(main.create_app()) as client:
        with SessionLocal() as db:
            db.add(
                VideoStream(
                    name="项目部门口",
                    stream_url="rtsp://example/1",
                    camera_id=camera_id,
                    location_name="研发中心 3F 项目部",
                )
            )
            image = Image(image_url="/data/frames/f.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            # Three crops seconds apart and deliberately no upsert_crop call: this is the lag.
            for offset in (0, 2, 4):
                crop = PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/{offset}.jpg",
                    bbox={"label": "person"},
                    camera_id=camera_id,
                    captured_at=captured + timedelta(seconds=offset),
                )
                db.add(crop)
                db.commit()
                db.refresh(crop)
                crop_ids.append(crop.id)
            db.commit()
            assert db.query(PersonObservationIndex).count() == 0

        response = client.post(
            "/api/reid/search",
            files={"file": ("query.jpg", b"bytes", "image/jpeg")},
        )

    assert response.status_code == 200
    items = response.json()["items"]
    # One visit, not three unplaceable frames.
    assert len(items) == 1
    assert items[0]["frame_count"] == 3
    assert items[0]["camera_name"] == "项目部门口"
    assert items[0]["location_name"] == "研发中心 3F 项目部"
    assert items[0]["first_seen"].startswith("2026-08-24T11:04:00")


def test_query_tracklet_keeps_only_nearby_identity_consistent_frames(monkeypatch, tmp_path):
    load_app(
        monkeypatch,
        tmp_path,
        "test-reid-query-tracklet",
        REID_QUERY_TRACKLET_FRAMES="3",
        REID_QUERY_TRACKLET_WINDOW_SECONDS="30",
        REID_QUERY_TRACKLET_IDENTITY_THRESHOLD="0.78",
    )

    from app.config.settings import get_settings
    from app.db.session import SessionLocal, init_db
    from app.models.media import Image, PersonCrop
    from app.services.reid_index import ReidIndexService
    from app.services.vector_index import MilvusVectorIndex

    init_db()
    camera_id = uuid.uuid4()
    captured = datetime(2026, 8, 24, 12, 0, 0)
    crop_dir = get_settings().data_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    ids: dict[str, uuid.UUID] = {}

    with SessionLocal() as db:
        image = Image(image_url="/data/frames/query.jpg", source_type="stream_frame")
        db.add(image)
        db.flush()
        for name, offset, confidence in (
            ("before", -15, 0.82),
            ("query", 0, 0.80),
            ("stranger", 5, 0.99),
            ("after", 15, 0.91),
            ("far", 90, 0.95),
        ):
            (crop_dir / f"{name}.jpg").write_bytes(b"image")
            item = PersonCrop(
                image_id=image.id,
                crop_url=f"/data/crops/{name}.jpg",
                bbox={"label": "person", "confidence": confidence, "width": 80, "height": 220},
                camera_id=camera_id,
                captured_at=captured + timedelta(seconds=offset),
            )
            db.add(item)
            db.flush()
            ids[name] = item.id
        db.commit()
        query = db.get(PersonCrop, ids["query"])
        assert query is not None

        vectors = {
            ids["before"]: [0.98, 0.02],
            ids["query"]: [1.0, 0.0],
            ids["stranger"]: [0.0, 1.0],
            ids["after"]: [0.96, 0.04],
            ids["far"]: [1.0, 0.0],
        }
        monkeypatch.setattr(
            MilvusVectorIndex,
            "fetch_vectors",
            lambda self, object_type, crop_ids: {
                crop_id: vectors[crop_id] for crop_id in crop_ids
            },
        )

        gallery = ReidIndexService(db, get_settings()).query_tracklet(query)

    assert [item.id for item in gallery] == [
        ids["query"],
        ids["before"],
        ids["after"],
    ]
