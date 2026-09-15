import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from test_reid import load_app


@pytest.fixture
def semantic_app(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "semantic",
        SEMANTIC_SEARCH_ENABLED="true",
        SEMANTIC_SEARCH_MIN_SCORE="0.25",
        MILVUS_ENABLED="true",
        MILVUS_VISUAL_COLLECTION_PREFIX="test_qwen_v1",
        VISUAL_EMBEDDING_PROVIDER="qwen3_vl_http",
        VISUAL_EMBEDDING_MODEL="Qwen/Qwen3-VL-Embedding-2B",
        VISUAL_EMBEDDING_DIM="2048",
        VISUAL_EMBEDDING_SERVICE_URL="http://embedding.invalid",
        VECTOR_INDEX_ON_INGEST="false",
        REID_ENABLED="false",
        VLM_PROVIDER="none",
    )
    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.models.vectors import VLEmbedding
    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    query = Mock(return_value=[])
    monkeypatch.setattr(MilvusVectorIndex, "search_text_for_objects", query)
    settings = get_settings()

    def add_crop(content=b"sample", indexed=True, **fields):
        crop_id = uuid.uuid4()
        filename = settings.data_dir / f"{crop_id}.jpg"
        filename.write_bytes(content)
        with SessionLocal() as db:
            image = Image(image_url="/data/original.jpg", source_type="upload")
            db.add(image)
            db.flush()
            crop = PersonCrop(
                id=crop_id,
                image_id=image.id,
                crop_url=f"/data/{filename.name}",
                bbox={"label": "person"},
                **fields,
            )
            db.add(crop)
            if indexed:
                db.add(VLEmbedding(
                    object_type="person_crop",
                    object_id=crop_id,
                    embedding_model=settings.visual_embedding_model,
                    embedding_dim=2048,
                ))
            db.commit()
        return crop_id

    with TestClient(main.create_app()) as client:
        yield SimpleNamespace(
            client=client, query=query, settings=settings, add_crop=add_crop,
            hit=VectorSearchHit, db=SessionLocal,
        )


def test_semantic_returns_candidates_without_changing_strict_labels(semantic_app):
    state = semantic_app
    crop_id = state.add_crop(person_id_source="manual")
    state.query.return_value = [state.hit(crop_id, 0.37)]
    strict = state.client.post("/api/search/person-crops", json={"query": "穿红衣服的男子"})
    assert strict.status_code == 200
    assert strict.json()["items"] == []
    state.query.assert_not_called()
    response = state.client.post(
        "/api/search/semantic/person-crops", json={"query": "穿红衣服的男子"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "semantic"
    assert payload["items"][0]["match_type"] == "semantic_candidate"
    assert payload["items"][0]["person_id"] is None
    assert payload["items"][0]["score"] == 0.37
    assert "未逐项核验" in payload["notice"]
    from app.models.media import PersonCrop
    with state.db() as db:
        crop = db.get(PersonCrop, crop_id)
        assert crop.attributes is None
        assert crop.person_id is None
        assert crop.person_id_source == "manual"


def test_filters_restrict_vector_scope_and_defensively_filter_results(semantic_app):
    state = semantic_app
    camera_id, location_id = uuid.uuid4(), uuid.uuid4()
    captured_at = datetime(2026, 1, 1, 12, tzinfo=UTC)
    wanted = state.add_crop(camera_id=camera_id, location_id=location_id, captured_at=captured_at)
    other_camera = state.add_crop(camera_id=uuid.uuid4(), captured_at=captured_at)
    no_time = state.add_crop(camera_id=camera_id, location_id=location_id)
    state.query.return_value = [
        state.hit(other_camera, 0.99), state.hit(no_time, 0.98), state.hit(wanted, 0.5)
    ]
    response = state.client.post("/api/search/semantic/person-crops", json={
        "query": "红衣", "top_k": 1,
        "filters": {
            "camera_id": str(camera_id), "location_id": str(location_id),
            "start_time": "2026-01-01T12:00:00Z", "end_time": "2026-01-01T12:00:00Z",
        },
    })
    assert response.status_code == 200
    assert [item["crop_id"] for item in response.json()["items"]] == [str(wanted)]
    assert state.query.call_args.args[-1] == [wanted]


def test_duplicate_crops_collapse_without_merging_cameras(semantic_app):
    state = semantic_app
    camera_id = uuid.uuid4()
    first = state.add_crop(camera_id=camera_id)
    second = state.add_crop(camera_id=camera_id)
    other_camera = state.add_crop(camera_id=uuid.uuid4())
    state.query.return_value = [
        state.hit(second, 0.5), state.hit(first, 0.6), state.hit(other_camera, 0.4)
    ]
    payload = state.client.post(
        "/api/search/semantic/person-crops", json={"query": "红衣"}
    ).json()
    assert len(payload["items"]) == 2
    assert payload["items"][0]["crop_id"] == str(first)
    assert payload["items"][0]["duplicate_crop_ids"] == [str(second)]


def test_no_score_threshold_matches_is_an_honest_empty(semantic_app):
    state = semantic_app
    crop_id = state.add_crop()
    state.query.return_value = [state.hit(crop_id, 0.17)]
    response = state.client.post(
        "/api/search/semantic/person-crops", json={"query": "穿宇航服的人"}
    )
    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["candidates_examined"] == 1


@pytest.mark.parametrize("failure", ["embedding", "milvus", "missing_vectors"])
def test_backend_errors_are_not_disguised_as_empty_results(semantic_app, failure):
    from app.services.embeddings import EmbeddingRuntimeError
    from app.services.vector_index import VectorIndexError
    state = semantic_app
    state.add_crop()
    if failure != "missing_vectors":
        state.query.side_effect = (
            EmbeddingRuntimeError if failure == "embedding" else VectorIndexError
        )("internal credential-bearing details must not escape")
    response = state.client.post("/api/search/semantic/person-crops", json={"query": "红衣"})
    assert response.status_code == 503
    assert "internal credential" not in response.text
    assert "items" not in response.json()


@pytest.mark.parametrize("query", ["", "   ", "字" * 2049])
def test_invalid_queries_never_reach_embedding(semantic_app, query):
    response = semantic_app.client.post(
        "/api/search/semantic/person-crops", json={"query": query}
    )
    assert response.status_code == 422
    semantic_app.query.assert_not_called()


def test_missing_current_model_index_is_reported(semantic_app):
    state = semantic_app
    state.add_crop(indexed=False)
    response = state.client.post("/api/search/semantic/person-crops", json={"query": "红衣"})
    assert response.status_code == 503
    assert "补建历史向量" in response.json()["detail"]
    state.query.assert_not_called()


def test_empty_filtered_scope_does_not_query_vectors(semantic_app):
    state = semantic_app
    state.add_crop()
    response = state.client.post("/api/search/semantic/person-crops", json={
        "query": "红衣", "filters": {"camera_id": str(uuid.uuid4())},
    })
    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["scope_crops"] == 0
    state.query.assert_not_called()


def test_disabled_rollback_and_clip_are_not_silently_enabled(semantic_app):
    state = semantic_app
    state.settings.semantic_search_enabled = False
    assert state.client.post(
        "/api/search/semantic/person-crops", json={"query": "红衣"}
    ).status_code == 503
    assert state.client.get("/api/search/semantic/status").json()["enabled"] is False
    state.settings.semantic_search_enabled = True
    state.settings.visual_embedding_provider = "sentence_transformers"
    assert state.client.post(
        "/api/search/semantic/person-crops", json={"query": "红衣"}
    ).status_code == 503
    state.query.assert_not_called()


def test_invalid_time_or_extra_filters_are_explicit_errors(semantic_app):
    for filters in [
        {"start_time": "2026-01-02T00:00:00Z", "end_time": "2026-01-01T00:00:00"},
        {"extra": {"hat": True}},
    ]:
        response = semantic_app.client.post(
            "/api/search/semantic/person-crops", json={"query": "红衣", "filters": filters}
        )
        assert response.status_code == 400
    semantic_app.query.assert_not_called()


def test_oversized_scope_requires_filters(semantic_app):
    state = semantic_app
    state.settings.semantic_search_max_scope = 1
    state.add_crop()
    state.add_crop(content=b"different")
    response = state.client.post("/api/search/semantic/person-crops", json={"query": "红衣"})
    assert response.status_code == 400
    assert "缩小" in response.json()["detail"]
    state.query.assert_not_called()


def test_status_and_partial_coverage_are_visible(semantic_app):
    state = semantic_app
    wanted = state.add_crop()
    state.add_crop(indexed=False)
    status = state.client.get("/api/search/semantic/status").json()
    assert status["total_crops"] == 2
    assert status["indexed_crops"] == 1
    assert status["labeled_crops"] == 0
    assert status["attributes_enabled"] is False
    state.query.return_value = [state.hit(wanted, 0.5)]
    result = state.client.post(
        "/api/search/semantic/person-crops", json={"query": "红衣"}
    ).json()
    assert "1/2" in result["notice"]


def test_missing_and_out_of_root_files_are_not_returned(semantic_app):
    state = semantic_app
    crop_id = state.add_crop()
    from app.models.media import PersonCrop
    for url in ["/data/missing.jpg", "/data/../../etc/passwd", "https://outside.invalid/x.jpg"]:
        with state.db() as db:
            db.get(PersonCrop, crop_id).crop_url = url
            db.commit()
        state.query.return_value = [state.hit(crop_id, 0.9)]
        response = state.client.post(
            "/api/search/semantic/person-crops", json={"query": "红衣"}
        )
        assert response.status_code == 200
        assert response.json()["items"] == []


def test_milvus_uuid_filter_is_applied_before_top_k(semantic_app, monkeypatch):
    from app.services.vector_index import MilvusVectorIndex
    index = MilvusVectorIndex(semantic_app.settings)
    collection = Mock()
    collection.search.return_value = [[]]
    monkeypatch.setattr(index, "_collection", lambda target: collection)
    object_id = uuid.uuid4()
    index._search_vector("person_crop", [0.5], 7, object_ids=[object_id])
    arguments = collection.search.call_args.kwargs
    assert arguments["expr"] == f'object_id in ["{object_id}"]'
    assert arguments["limit"] == 7
    index._search_vector("reid_person_crop", [0.5], 7)
    assert "expr" not in collection.search.call_args.kwargs
    collection.search.reset_mock()
    assert index._search_vector("person_crop", [0.5], 7, object_ids=[]) == []
    collection.search.assert_not_called()
