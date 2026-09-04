import importlib
import math
import os
import sys
import threading
import types
import uuid
from base64 import b64encode
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text


def load_app(monkeypatch, tmp_path, name: str):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / f'{name}.db'}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PERSON_DETECTOR", "whole_frame")
    for module_name in list(sys.modules):
        if module_name == "main" or module_name.startswith("app."):
            sys.modules.pop(module_name)
    return importlib.import_module("main")


def test_health(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test")
    with TestClient(main.create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_spa_serves_index_for_client_routes(monkeypatch, tmp_path):
    """Every client route must resolve to index.html; the router runs in history mode."""

    main = load_app(monkeypatch, tmp_path, "test-spa-routes")
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id=app></div>", encoding="utf-8")
    (dist / "assets" / "index-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "favicon.ico").write_text("icon", encoding="utf-8")
    monkeypatch.setattr(main, "FRONTEND_DIST", dist)

    with TestClient(main.create_app()) as client:
        for path in ("/", "/search", "/observations", "/faces", "/chat-ui", "/faces?person_id=1"):
            response = client.get(path)
            assert response.status_code == 200, path
            assert "<div id=app>" in response.text, path
            assert response.headers["cache-control"] == "no-store", path

        # A real file under dist wins over the catch-all.
        assert client.get("/favicon.ico").text == "icon"
        assert client.get("/assets/index-abc123.js").status_code == 200
        # The API must not be swallowed by the catch-all.
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/persons").status_code == 200


def test_spa_reports_missing_build_instead_of_404(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-spa-missing")
    monkeypatch.setattr(main, "FRONTEND_DIST", tmp_path / "not-built")

    with TestClient(main.create_app()) as client:
        response = client.get("/")
        assert response.status_code == 503
        assert "npm ci && npm run build" in response.text
        # The API keeps working even without a frontend bundle.
        assert client.get("/health").json() == {"status": "ok"}


def test_basic_auth_protects_app_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_BASIC_AUTH_USERNAME", "viewer")
    monkeypatch.setenv("APP_BASIC_AUTH_PASSWORD", "secret")
    main = load_app(monkeypatch, tmp_path, "test-basic-auth")
    token = b64encode(b"viewer:secret").decode("ascii")

    with TestClient(main.create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 401
        assert client.get("/api/streams").status_code == 401
        assert client.get("/data/frames/missing.jpg").status_code == 401

        response = client.get(
            "/api/streams",
            headers={"Authorization": f"Basic {token}"},
        )

    assert response.status_code == 200
    assert response.json() == []


def test_openai_compatible_embeddings_uses_bearer_without_basic_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_BASIC_AUTH_USERNAME", "viewer")
    monkeypatch.setenv("APP_BASIC_AUTH_PASSWORD", "secret")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "dashscope_multimodal")
    monkeypatch.setenv("VISUAL_EMBEDDING_MODEL", "qwen3-vl-embedding")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2")
    monkeypatch.setenv("VISUAL_EMBEDDING_SERVICE_API_KEY", "dashscope-token")
    main = load_app(monkeypatch, tmp_path, "test-openai-embedding")

    def fake_embed_text(self, text):
        assert text == "白色衣服"
        return [0.2, 0.8]

    monkeypatch.setattr(
        "app.services.embeddings.VisualEmbeddingService.embed_text",
        fake_embed_text,
    )

    with TestClient(main.create_app()) as client:
        unauthorized = client.post(
            "/v1/embeddings",
            json={"model": "qwen3-vl-embedding", "input": "白色衣服"},
        )
        response = client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer dashscope-token"},
            json={"model": "qwen3-vl-embedding", "input": "白色衣服"},
        )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.2, 0.8],
            }
        ],
        "model": "qwen3-vl-embedding",
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }


def test_openai_compatible_embeddings_accepts_image_data_url(monkeypatch, tmp_path):
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "dashscope_multimodal")
    monkeypatch.setenv("VISUAL_EMBEDDING_MODEL", "qwen3-vl-embedding")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2")
    main = load_app(monkeypatch, tmp_path, "test-openai-embedding-image")

    captured = {}

    def fake_embed_image(self, image_path):
        captured["suffix"] = image_path.suffix
        captured["bytes"] = image_path.read_bytes()
        return [0.4, 0.6]

    monkeypatch.setattr(
        "app.services.embeddings.VisualEmbeddingService.embed_image",
        fake_embed_image,
    )

    with TestClient(main.create_app()) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "input": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,aW1n",
                        },
                    }
                ]
            },
        )

    assert response.status_code == 200
    assert response.json()["data"][0]["embedding"] == [0.4, 0.6]
    assert captured == {"suffix": ".png", "bytes": b"img"}


def test_visual_embedding_endpoints_bypass_basic_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_BASIC_AUTH_USERNAME", "viewer")
    monkeypatch.setenv("APP_BASIC_AUTH_PASSWORD", "secret")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "dashscope_multimodal")
    monkeypatch.setenv("VISUAL_EMBEDDING_MODEL", "qwen3-vl-embedding")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2")
    monkeypatch.setenv("VISUAL_EMBEDDING_SERVICE_API_KEY", "dashscope-token")
    main = load_app(monkeypatch, tmp_path, "test-embedding-auth-exempt")

    def fake_embed_image(self, image_path):
        assert image_path.read_bytes() == b"img"
        return [0.7, 0.3]

    monkeypatch.setattr(
        "app.services.embeddings.VisualEmbeddingService.embed_image",
        fake_embed_image,
    )

    payload = {"image_base64": "aW1n", "image_filename": "query.jpg"}
    with TestClient(main.create_app()) as client:
        browser_route = client.get("/api/streams")
        missing_key = client.post("/api/embeddings/image-vector", json=payload)
        image_vector = client.post(
            "/api/embeddings/image-vector",
            headers={"Authorization": "Bearer dashscope-token"},
            json=payload,
        )
        visual = client.post(
            "/api/embeddings/visual",
            headers={"X-API-Key": "dashscope-token"},
            json=payload,
        )

    assert browser_route.status_code == 401
    assert browser_route.headers["WWW-Authenticate"].startswith("Basic")
    assert missing_key.status_code == 401
    assert "WWW-Authenticate" not in missing_key.headers
    assert image_vector.status_code == 200
    assert image_vector.json()["embedding"] == [0.7, 0.3]
    assert image_vector.json()["input_type"] == "image"
    assert visual.status_code == 200
    assert visual.json()["embedding"] == [0.7, 0.3]


def test_vector_index_on_ingest_uses_background_queue(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_INDEX_ON_INGEST", "true")
    monkeypatch.setenv("VECTOR_INDEX_ON_INGEST_BACKGROUND", "true")
    monkeypatch.setenv("MILVUS_ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    main = load_app(monkeypatch, tmp_path, "test-vector-index-queue")
    from app.db.session import SessionLocal
    from app.services.media import MediaService

    enqueued = []

    class FakeVectorIndexQueue:
        def start(self, settings=None):
            return None

        def stop(self):
            return None

        def target_enabled(self, target, settings):
            return target == "image"

        def enqueue_in_session(self, db, target, image_id, settings):
            assert target == "image"
            enqueued.append(image_id)
            return True

        def wake(self, settings=None):
            return None

    def fail_if_sync_index_is_used(*args, **kwargs):
        raise AssertionError("synchronous vector indexing should not run")

    monkeypatch.setattr(
        "app.services.vector_index_queue.vector_index_queue",
        FakeVectorIndexQueue(),
    )
    monkeypatch.setattr(main, "vector_index_queue", FakeVectorIndexQueue())
    monkeypatch.setattr(
        "app.services.vector_index.VectorIndexingService",
        fail_if_sync_index_is_used,
    )

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            image = MediaService(db, main.get_settings()).create_image_from_url(
                "/data/frames/a.jpg",
                source_type="upload",
            )

    assert enqueued == [image.id]


def test_vector_index_queue_persists_jobs(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_INDEX_ON_INGEST", "true")
    monkeypatch.setenv("VECTOR_INDEX_ON_INGEST_BACKGROUND", "true")
    monkeypatch.setenv("MILVUS_ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("VECTOR_INDEX_BACKGROUND_IDLE_SECONDS", "60")
    main = load_app(monkeypatch, tmp_path, "test-vector-index-persistent-queue")
    from app.db.session import SessionLocal
    from app.models.vectors import VectorIndexJob
    from app.services.media import MediaService
    from app.services.vector_index_queue import vector_index_queue

    class FakeMainVectorIndexQueue:
        def start(self, settings=None):
            return None

        def stop(self):
            return None

    monkeypatch.setattr(main, "vector_index_queue", FakeMainVectorIndexQueue())
    monkeypatch.setattr(vector_index_queue, "start", lambda settings=None: None)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            image = MediaService(db, main.get_settings()).create_image_from_url(
                "/data/frames/persist.jpg",
                source_type="upload",
            )
            jobs = list(db.query(VectorIndexJob).all())

        vector_index_queue.enqueue_image(image.id)
        with SessionLocal() as db:
            jobs_after_duplicate = list(db.query(VectorIndexJob).all())

    assert len(jobs) == 1
    assert jobs[0].target == "image"
    assert jobs[0].object_id == image.id
    assert jobs[0].status == "pending"
    assert len(jobs_after_duplicate) == 1


def test_vector_index_queue_processes_persisted_job(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_INDEX_ON_INGEST", "true")
    monkeypatch.setenv("VECTOR_INDEX_ON_INGEST_BACKGROUND", "true")
    monkeypatch.setenv("MILVUS_ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("VECTOR_INDEX_BACKGROUND_BATCH_SIZE", "4")
    monkeypatch.setenv("VECTOR_INDEX_BACKGROUND_IDLE_SECONDS", "60")
    main = load_app(monkeypatch, tmp_path, "test-vector-index-persistent-process")
    from app.db.session import SessionLocal
    from app.models.vectors import VectorIndexJob
    from app.services.media import MediaService
    from app.services.vector_index_queue import vector_index_queue

    class FakeMainVectorIndexQueue:
        def start(self, settings=None):
            return None

        def stop(self):
            return None

    indexed = []

    class FakeIndex:
        def flush(self, target):
            return None

    class FakeVectorIndexingService:
        def __init__(self, db, settings):
            self.index = FakeIndex()

        def index_image(self, image, flush=True):
            indexed.append(image.id)

        def index_crop(self, crop, flush=True):
            indexed.append(crop.id)

    monkeypatch.setattr(main, "vector_index_queue", FakeMainVectorIndexQueue())
    monkeypatch.setattr(vector_index_queue, "start", lambda settings=None: None)
    monkeypatch.setattr(
        "app.services.vector_index_queue.VectorIndexingService",
        FakeVectorIndexingService,
    )

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            image = MediaService(db, main.get_settings()).create_image_from_url(
                "/data/frames/process.jpg",
                source_type="upload",
            )
        jobs = vector_index_queue._claim_jobs(main.get_settings())
        with SessionLocal() as db:
            indexer = FakeVectorIndexingService(db, main.get_settings())
            for job in jobs:
                vector_index_queue._index_job(db, indexer, job)
                vector_index_queue._mark_done(job.id)
            remaining = list(db.query(VectorIndexJob).all())

    assert indexed == [image.id]
    assert remaining == []


def test_person_chat_and_search_flow(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-flow")
    with TestClient(main.create_app()) as client:
        person_response = client.post("/api/persons", json={"name": "张三"})
        assert person_response.status_code == 200
        assert person_response.json()["name"] == "张三"

        search_response = client.post(
            "/api/search/person-crops",
            json={"query": "双马尾辫", "top_k": 20, "filters": {}},
        )
        assert search_response.status_code == 200
        assert search_response.json() == {"items": []}

        chat_response = client.post("/api/chat", json={"message": "今天有多少人？"})
        assert chat_response.status_code == 200
        payload = chat_response.json()
        assert payload["tool_name"] == "count_events"
        assert payload["data"]["counting_event_count"] == 0


def test_image_vector_endpoint_returns_visual_embedding(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-image-vector")
    from app.services.embeddings import VisualEmbeddingService

    captured = {}

    def fake_embed_image(self, image_path):
        captured["filename"] = image_path.name
        captured["bytes"] = image_path.read_bytes()
        return [0.1, 0.2, 0.3]

    def fake_embed_text(self, text):
        captured["text"] = text
        return [0.4, 0.5, 0.6]

    monkeypatch.setattr(VisualEmbeddingService, "embed_image", fake_embed_image)
    monkeypatch.setattr(VisualEmbeddingService, "embed_text", fake_embed_text)

    with TestClient(main.create_app()) as client:
        image_response = client.post(
            "/api/embeddings/image-vector",
            json={"image_base64": "aW1hZ2U=", "image_filename": "crop.jpg"},
        )
        text_response = client.post(
            "/api/embeddings/image-vector",
            json={"text": "黑衣背包的人"},
        )

    assert image_response.status_code == 200
    assert image_response.json()["embedding"] == [0.1, 0.2, 0.3]
    assert image_response.json()["dim"] == 3
    assert image_response.json()["input_type"] == "image"
    assert text_response.status_code == 200
    assert text_response.json()["embedding"] == [0.4, 0.5, 0.6]
    assert text_response.json()["input_type"] == "text"
    assert captured == {
        "filename": "query.jpg",
        "bytes": b"image",
        "text": "黑衣背包的人",
    }


def test_vlm_structured_analysis_endpoint_returns_attributes(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-vlm-structured")
    from app.services.vlm import VLMStructuredAnalysisService

    captured = {}

    def fake_analyze_image(self, image_path, object_type, bbox=None):
        captured["filename"] = image_path.name
        captured["bytes"] = image_path.read_bytes()
        captured["object_type"] = object_type
        captured["bbox"] = bbox
        return {
            "object_type": "person",
            "clothing": {"upper_color": "black"},
            "objects": {"backpack": True},
        }

    monkeypatch.setattr(VLMStructuredAnalysisService, "analyze_image", fake_analyze_image)

    with TestClient(main.create_app()) as client:
        response = client.post(
            "/api/vlm/structured-analysis",
            json={
                "image_base64": "data:image/jpeg;base64,aW1hZ2U=",
                "image_filename": "person.jpg",
                "object_type": "person",
                "label_language": "zh",
                "bbox": {"x1": 1, "y1": 2, "x2": 30, "y2": 40, "label": "person"},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["attributes"]["clothing"]["upper_color"] == "black"
    assert payload["attributes"]["objects"]["backpack"] is True
    assert payload["label_language"] == "zh"
    assert payload["labels"]["对象类型"] == "人员"
    assert payload["labels"]["上衣颜色"] == "黑色"
    assert payload["labels"]["背包"] == "是"
    assert captured["filename"] == "query.jpg"
    assert captured["bytes"] == b"image"
    assert captured["object_type"] == "person"
    assert captured["bbox"]["label"] == "person"


def test_vlm_scene_summary_endpoint_returns_tags(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-vlm-scene-summary")
    from app.services.vlm import VLMSceneSummaryService

    captured = {}

    def fake_summarize_image(self, image_path, context=None):
        captured["filename"] = image_path.name
        captured["bytes"] = image_path.read_bytes()
        captured["context"] = context
        return {
            "title": "白色SUV内两名男子,驾驶员未系安全带",
            "description": (
                "夜间道路,白色宝马SUV正前方行驶,驾驶员穿白色长袖握方向盘未系安全带,"
                "副驾驶穿黑色上衣看手机,车灯开启"
            ),
            "tags": {
                "target": ["vehicle", "suv", "bmw"],
                "attribute": [
                    "white",
                    "long_sleeve",
                    "black_upper",
                    "headlights_on",
                    "blue_plate",
                ],
                "behavior": ["driving", "phone_calling"],
                "status": ["unbelted", "front"],
                "scene": ["road", "night"],
            },
            "persons": [
                {
                    "role": "driver",
                    "tags": {
                        "attribute": ["white", "long_sleeve"],
                        "behavior": ["driving"],
                        "status": ["unbelted", "front"],
                    },
                },
                {
                    "role": "passenger",
                    "tags": {
                        "attribute": ["black_upper"],
                        "behavior": ["phone_calling"],
                        "status": ["front"],
                    },
                },
            ],
            "count": {"persons": 2, "vehicles": 1},
            "confidence": 0.86,
        }

    monkeypatch.setattr(VLMSceneSummaryService, "summarize_image", fake_summarize_image)

    with TestClient(main.create_app()) as client:
        response = client.post(
            "/api/vlm/scene-summary",
            json={
                "image_base64": "data:image/jpeg;base64,aW1hZ2U=",
                "image_filename": "traffic.jpg",
                "context": "traffic violation",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "白色SUV内两名男子,驾驶员未系安全带"
    assert payload["tags"]["target"] == ["vehicle", "suv", "bmw"]
    assert payload["tags"]["status"] == ["unbelted", "front"]
    assert payload["persons"][0]["role"] == "driver"
    assert payload["persons"][0]["tags"]["status"] == ["unbelted", "front"]
    assert payload["count"] == {"persons": 2, "vehicles": 1}
    assert payload["confidence"] == 0.86
    assert captured == {
        "filename": "query.jpg",
        "bytes": b"image",
        "context": "traffic violation",
    }


def test_count_chat_reports_detected_crops(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-count-crops")
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            image = Image(
                image_url="/data/frames/a.jpg",
                source_type="stream_frame",
                created_at=now,
            )
            db.add(image)
            db.commit()
            db.refresh(image)
            db.add(
                PersonCrop(
                    image_id=image.id,
                    crop_url="/data/crops/a.jpg",
                    bbox={"x": 1, "y": 2, "width": 3, "height": 4, "label": "person"},
                    created_at=now,
                )
            )
            db.commit()

        chat_response = client.post("/api/chat", json={"message": "今天有多少人？"})

    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["tool_name"] == "count_events"
    assert payload["data"]["person_crop_count"] == 1
    assert payload["data"]["image_count"] == 1
    assert payload["data"]["counting_event_count"] == 0
    assert "今天检测到人体裁剪 1 个" in payload["answer"]


def test_chat_reports_stream_status(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-chat-stream-status")
    from app.db.session import SessionLocal
    from app.models.media import VideoStream

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            db.add_all(
                [
                    VideoStream(name="入口", stream_url="rtsp://example/1", status="running"),
                    VideoStream(name="出口", stream_url="rtsp://example/2", status="stopped"),
                ]
            )
            db.commit()

        response = client.post("/api/chat", json={"message": "现在有几个视频流在运行？"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_name"] == "stream_status"
    assert payload["data"]["total"] == 2
    assert payload["data"]["running"] == 1
    assert payload["data"]["items"][0]["name"] in {"入口", "出口"}


def test_chat_reports_face_summary(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-chat-face-summary")
    from app.db.session import SessionLocal
    from app.models.persons import Person
    from app.models.vectors import FaceEmbedding

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            person = Person(name="张三", employee_no="E001")
            db.add(person)
            db.commit()
            db.refresh(person)
            db.add(FaceEmbedding(person_id=person.id, embedding=[1.0, 0.0], face_model="test"))
            db.commit()

        response = client.post("/api/chat", json={"message": "人脸库有多少人？"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_name"] == "face_summary"
    assert payload["data"]["person_count"] == 1
    assert payload["data"]["face_count"] == 1
    assert payload["data"]["items"][0]["person_name"] == "张三"


def test_chat_face_image_search_uses_last_uploaded_image_context(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-chat-face-image-search")
    from app.db.session import SessionLocal
    from app.models.media import Image
    from app.models.persons import Person
    from app.schemas.persons import FaceMatchItem, FaceSearchResponse
    from app.services.faces import FaceRecognitionService

    captured = {}
    person_id = uuid.uuid4()

    def fake_search_image(self, image, top_k=5, min_similarity=None, allow_fallback=True):
        captured["image_id"] = image.id
        captured["top_k"] = top_k
        return FaceSearchResponse(
            image_id=image.id,
            face_bbox={"x": 1.0, "y": 2.0, "width": 32.0, "height": 32.0},
            matches=[
                FaceMatchItem(
                    person_id=person_id,
                    person_name="张三",
                    face_embedding_id=uuid.uuid4(),
                    similarity=0.93,
                    quality_score=0.88,
                )
            ],
        )

    monkeypatch.setattr(FaceRecognitionService, "search_image", fake_search_image)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            db.add(Person(id=person_id, name="张三", avatar_url="/data/uploads/zhangsan.jpg"))
            image = Image(image_url="/data/uploads/query.jpg", source_type="upload")
            db.add(image)
            db.commit()
            db.refresh(image)
            image_id = image.id

        response = client.post(
            "/api/chat",
            json={
                "message": "这张图片里的人脸是谁？",
                "context": {"last_image_id": str(image_id)},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_name"] == "face_image_search"
    assert payload["data"]["result_type"] == "known"
    assert payload["data"]["items"][0]["person_name"] == "张三"
    assert payload["data"]["items"][0]["score"] == 0.93
    assert payload["data"]["items"][0]["face_url"] == "/data/uploads/zhangsan.jpg"
    assert payload["data"]["items"][0]["avatar_url"] == "/data/uploads/zhangsan.jpg"
    assert captured["image_id"] == image_id
    assert captured["top_k"] == 5


def test_chat_returns_named_person_trajectory(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-chat-person-trajectory")
    from app.db.session import SessionLocal
    from app.models.events import RecognitionEvent
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.services.observation_index import ObservationIndexService

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            camera_id = uuid.uuid4()
            location_id = uuid.uuid4()
            person = Person(name="张三", department="研发部", phone="13800000000")
            image = Image(
                image_url="/data/frames/zhangsan.jpg",
                source_type="stream_frame",
                camera_id=camera_id,
                location_id=location_id,
            )
            db.add_all([person, image])
            db.commit()
            db.refresh(person)
            db.refresh(image)
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/zhangsan.jpg",
                bbox={"label": "person"},
                person_id=person.id,
                captured_at=now,
                camera_id=camera_id,
                location_id=location_id,
            )
            db.add(crop)
            db.commit()
            db.refresh(crop)
            db.add(
                RecognitionEvent(
                    image_id=image.id,
                    crop_id=crop.id,
                    person_id=person.id,
                    confidence=0.8,
                    similarity=0.92,
                    result_type="known",
                    recognized_at=now,
                    camera_id=camera_id,
                    location_id=location_id,
                )
            )
            ObservationIndexService(db, main.get_settings()).upsert_crop(crop)
            db.commit()

        response = client.post("/api/chat", json={"message": "张三今天在哪？"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_name"] == "person_trajectory"
    assert payload["tool_params"]["mode"] == "all"
    assert payload["data"]["person"]["name"] == "张三"
    assert payload["data"]["person"]["department"] == "研发部"
    assert payload["data"]["person"]["phone"] == "13800000000"
    assert {"key": "department", "label": "部门", "value": "研发部"} in payload["data"][
        "person"
    ]["tags"]
    assert payload["data"]["items"][0]["crop_url"] == "/data/crops/zhangsan.jpg"
    assert payload["data"]["items"][0]["match_source"] == "face"
    assert payload["data"]["items"][0]["location"]["location_id"] == str(location_id)
    assert payload["data"]["items"][0]["location"]["camera_id"] == str(camera_id)
    assert "研发部" in payload["answer"]
    assert "地点：" in payload["answer"]


def test_enroll_face_from_observation_crop_updates_wide_table(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-enroll-observation-crop")
    from app.db.session import SessionLocal
    from app.models.events import RecognitionEvent
    from app.models.media import Image, PersonCrop, PersonObservationIndex
    from app.models.persons import Person
    from app.services.faces import FaceCandidate, FaceRecognitionService

    def fake_best_candidate(self, image_url, allow_fallback=True):
        return FaceCandidate(
            embedding=[1.0, 0.0],
            bbox={"x": 1.0, "y": 2.0, "width": 24.0, "height": 24.0},
            quality_score=0.91,
            model="test-face",
        )

    monkeypatch.setattr(FaceRecognitionService, "_best_candidate", fake_best_candidate)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            person = Person(name="测试人员", department="测试部门")
            image = Image(
                image_url="/data/frames/person.jpg",
                source_type="stream_frame",
                captured_at=now,
            )
            db.add_all([person, image])
            db.commit()
            db.refresh(person)
            db.refresh(image)
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/person.jpg",
                bbox={"label": "person"},
                captured_at=now,
            )
            db.add(crop)
            db.commit()
            db.refresh(crop)
            person_id = person.id
            crop_id = crop.id

        response = client.post(f"/api/persons/{person_id}/faces/from-crop/{crop_id}")
        observations = client.get("/api/search/observations?query=测试人员&limit=10")
        trajectory = client.get(f"/api/persons/{person_id}/trajectory")

        with SessionLocal() as db:
            crop = db.get(PersonCrop, crop_id)
            event = db.query(RecognitionEvent).filter_by(crop_id=crop_id).one()
            row = db.query(PersonObservationIndex).filter_by(crop_id=crop_id).one()

    assert response.status_code == 200
    assert response.json()["person_id"] == str(person_id)
    assert crop.person_id == person_id
    assert event.person_id == person_id
    assert event.result_type == "known"
    assert row.person_id == person_id
    assert row.person_name == "测试人员"
    assert row.department == "测试部门"
    assert row.has_face_embedding is True
    assert observations.status_code == 200
    assert observations.json()["items"][0]["person_name"] == "测试人员"
    assert trajectory.status_code == 200
    assert trajectory.json()["items"][0]["crop_id"] == str(crop_id)
    assert trajectory.json()["items"][0]["person_name"] == "测试人员"


def test_chat_reports_person_today_work_time_from_face_library(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-chat-person-work-time")
    from app.db.session import SessionLocal
    from app.models.events import RecognitionEvent
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.services.observation_index import ObservationIndexService

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            first_time = now.replace(hour=8, minute=43, second=12, microsecond=0)
            later_time = now.replace(hour=17, minute=5, second=30, microsecond=0)
            camera_id = uuid.uuid4()
            location_id = uuid.uuid4()
            person = Person(
                name="张三",
                employee_no="E001",
                department="安保部",
                phone="13900000000",
            )
            image = Image(
                image_url="/data/frames/work-time.jpg",
                source_type="stream_frame",
                camera_id=camera_id,
                location_id=location_id,
            )
            db.add_all([person, image])
            db.commit()
            db.refresh(person)
            db.refresh(image)
            first_crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/work-first.jpg",
                bbox={"label": "person"},
                person_id=person.id,
                captured_at=first_time,
                camera_id=camera_id,
                location_id=location_id,
            )
            later_crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/work-later.jpg",
                bbox={"label": "person"},
                person_id=person.id,
                captured_at=later_time,
                camera_id=camera_id,
                location_id=location_id,
            )
            db.add_all([first_crop, later_crop])
            db.commit()
            db.refresh(first_crop)
            db.refresh(later_crop)
            db.add_all(
                [
                    RecognitionEvent(
                        image_id=image.id,
                        crop_id=later_crop.id,
                        person_id=person.id,
                        confidence=0.86,
                        similarity=0.9,
                        result_type="known",
                        recognized_at=later_time,
                        camera_id=camera_id,
                        location_id=location_id,
                    ),
                    RecognitionEvent(
                        image_id=image.id,
                        crop_id=first_crop.id,
                        person_id=person.id,
                        confidence=0.88,
                        similarity=0.93,
                        result_type="known",
                        recognized_at=first_time,
                        camera_id=camera_id,
                        location_id=location_id,
                    ),
                ]
            )
            db.commit()
            ObservationIndexService(db, main.get_settings()).upsert_crop(first_crop)
            ObservationIndexService(db, main.get_settings()).upsert_crop(later_crop)
            db.commit()

        response = client.post("/api/chat", json={"message": "张三今天几点上班？"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_name"] == "person_attendance"
    assert payload["tool_params"]["intent"] == "earliest"
    assert payload["tool_params"]["mode"] == "face"
    assert payload["data"]["person"]["employee_no"] == "E001"
    assert payload["data"]["person"]["department"] == "安保部"
    assert payload["data"]["person"]["phone"] == "13900000000"
    assert {"key": "employee_no", "label": "工号", "value": "E001"} in payload["data"][
        "person"
    ]["tags"]
    assert {"key": "department", "label": "部门", "value": "安保部"} in payload["data"][
        "person"
    ]["tags"]
    assert payload["data"]["source"] == "face"
    assert payload["data"]["appearance_count"] == 2
    assert payload["data"]["first_appearance"]["crop_url"] == "/data/crops/work-first.jpg"
    assert payload["data"]["first_appearance"]["location"]["location_id"] == str(location_id)
    assert payload["data"]["first_appearance"]["location"]["camera_id"] == str(camera_id)
    assert payload["data"]["items"][0]["match_source"] == "face"
    assert "08:43:12" in payload["answer"]
    assert "安保部" in payload["answer"]
    assert "地点：" in payload["answer"]
    assert "来源：人脸" in payload["answer"]


def test_chat_attendance_does_not_run_vector_fallback_by_default(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-chat-attendance-face-first")
    from app.db.session import SessionLocal
    from app.models.persons import Person
    from app.services.vector_index import MilvusVectorIndex

    def fail_vector_search(*_args, **_kwargs):
        raise AssertionError("attendance should not run vector trajectory by default")

    monkeypatch.setattr(MilvusVectorIndex, "search_image", fail_vector_search)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            db.add(Person(name="张三", employee_no="E001"))
            db.commit()

        response = client.post("/api/chat", json={"message": "张三今天几点上班？"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_name"] == "person_attendance"
    assert payload["tool_params"]["mode"] == "face"
    assert payload["answer"] == "没有查到 张三（张三 / E001 / active） 今天的到岗或出现记录。"


def test_chat_named_attendance_reports_missing_person_before_visual_fallback(
    monkeypatch,
    tmp_path,
):
    main = load_app(monkeypatch, tmp_path, "test-chat-attendance-missing-person")

    with TestClient(main.create_app()) as client:
        client.get("/health")
        response = client.post("/api/chat", json={"message": "张三今天几点上班？"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_name"] == "person_attendance"
    assert payload["answer"] == "没有匹配到人员。"


def test_chat_visual_count_routes_to_strict_structured_search(monkeypatch, tmp_path):
    monkeypatch.setenv("MILVUS_ENABLED", "true")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "qwen3_vl")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2")
    main = load_app(monkeypatch, tmp_path, "test-chat-visual-count")
    from app.schemas.media import SearchResponse, SearchResultItem
    from app.services.search import VisualSearchService

    captured = {}

    def fake_search(self, payload):
        captured["payload"] = payload
        return SearchResponse(
            items=[
                SearchResultItem(
                    crop_url="/data/crops/brown.jpg",
                    score=0.31,
                )
            ]
        )

    monkeypatch.setattr(VisualSearchService, "search", fake_search)

    with TestClient(main.create_app()) as client:
        response = client.post(
            "/api/chat",
            json={"message": "今天有几个穿褐色衣服的人？"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_name"] == "search_structured"
    assert payload["data"]["source"] == "structured_labels"
    assert "vector_status" not in payload["data"]
    assert [call["name"] for call in payload["tool_calls"]] == ["search_structured"]
    assert "返回 0 个" in payload["answer"]
    assert payload["data"]["items"] == []
    assert captured == {}



def test_chat_does_not_use_generic_clip_image_search(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-chat-image-search")
    from app.db.session import SessionLocal
    from app.models.media import Image
    from app.schemas.media import SearchResponse, SearchResultItem
    from app.services.search import VisualSearchService

    captured = {}

    def fake_search_by_image(self, payload):
        captured["payload"] = payload
        return SearchResponse(
            items=[
                SearchResultItem(
                    crop_url="/data/crops/similar.jpg",
                    score=0.88,
                )
            ]
        )

    monkeypatch.setattr(VisualSearchService, "search_by_image", fake_search_by_image)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            image = Image(image_url="/data/frames/query.jpg", source_type="upload")
            db.add(image)
            db.commit()
            db.refresh(image)
            image_id = image.id

        response = client.post(
            "/api/chat",
            json={
                "message": "找和刚才图片相似的人",
                "context": {"last_image_id": str(image_id)},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_name"] == "visual_text_search"
    assert payload["data"]["source"] == "label_keyword"
    assert payload["data"]["items"] == []
    assert captured == {}


def test_chat_structured_search_uses_person_crop_attributes(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-chat-structured-search")
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            image = Image(image_url="/data/frames/a.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            bald_crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/bald.jpg",
                bbox={"label": "person"},
                attributes={
                    "appearance": {"hair": "bald", "hair_confidence": 0.91, "hat": False},
                    "clothing": {"upper_color": "black"},
                },
            )
            short_hair_crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/short_hair.jpg",
                bbox={"label": "person"},
                attributes={
                    "appearance": {"hair": "short_hair", "hat": False},
                    "clothing": {"upper_color": "white"},
                },
            )
            low_confidence_crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/low_confidence_bald.jpg",
                bbox={"label": "person"},
                attributes={
                    "appearance": {"hair": "bald", "hair_confidence": 0.32, "hat": False},
                },
            )
            db.add_all([bald_crop, short_hair_crop, low_confidence_crop])
            db.commit()
            db.refresh(bald_crop)
            bald_crop_id = bald_crop.id

        response = client.post("/api/chat", json={"message": "找光头的人"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_name"] == "search_structured"
    assert payload["data"]["source"] == "structured_labels"
    assert len(payload["data"]["items"]) == 1
    assert payload["data"]["items"][0]["crop_id"] == str(bald_crop_id)
    assert payload["data"]["items"][0]["attributes"]["appearance"]["hair"] == "bald"
    assert payload["data"]["conditions"] == [
        {"field": "hair", "values": ["bald", "shaved"]}
    ]
    assert [call["name"] for call in payload["tool_calls"]] == ["search_structured"]


def test_vlm_structured_person_crop_analysis_updates_attributes(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("VLM_BASE_URL", "http://vlm.local/v1")
    monkeypatch.setenv("VLM_MODEL", "test-vlm")
    main = load_app(monkeypatch, tmp_path, "test-vlm-person-attributes")

    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services import vlm

    class FakeVLMResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return (
                b'{"choices":[{"message":{"content":"{'
                b'\\"object_type\\":\\"person\\",'
                b'\\"appearance\\":{\\"hair\\":\\"short_hair\\",\\"hat\\":true,'
                b'\\"glasses\\":false,\\"hat_confidence\\":0.93},'
                b'\\"clothing\\":{\\"upper_color\\":\\"red\\",'
                b'\\"upper_color_confidence\\":0.88},'
                b'\\"objects\\":{\\"backpack\\":true},'
                b'\\"behavior\\":{\\"smoking\\":false},'
                b'\\"confidence\\":0.9}"} }]}'
            )

    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = req.data.decode("utf-8")
        return FakeVLMResponse()

    monkeypatch.setattr(vlm.request, "urlopen", fake_urlopen)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        data_dir = tmp_path / "data"
        (data_dir / "crops").mkdir(parents=True, exist_ok=True)
        (data_dir / "crops" / "person.jpg").write_bytes(b"person-image")
        with SessionLocal() as db:
            image = Image(image_url="/data/frames/source.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/person.jpg",
                bbox={"x": 1, "y": 2, "width": 80, "height": 180, "label": "person"},
            )
            db.add(crop)
            db.commit()
            db.refresh(crop)
            crop_id = crop.id

        response = client.post(f"/api/attributes/person-crops/{crop_id}/analyze")
        search_response = client.post("/api/chat", json={"message": "找红衣戴帽的人"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["crop_id"] == str(crop_id)
    assert payload["attributes"]["object_type"] == "person"
    assert payload["attributes"]["appearance"]["hat"] is True
    assert payload["attributes"]["clothing"]["upper_color"] == "red"
    assert payload["attributes"]["top_color"] == "red"
    assert payload["attributes"]["has_hat"] is True
    assert captured["url"] == "http://vlm.local/v1/chat/completions"
    assert '"response_format": {"type": "json_object"}' in captured["payload"]

    assert search_response.status_code == 200
    search_payload = search_response.json()
    assert search_payload["tool_name"] == "search_structured"
    assert search_payload["data"]["items"][0]["crop_id"] == str(crop_id)


def test_vlm_structured_vehicle_upload_analysis(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("VLM_BASE_URL", "http://vlm.local/v1")
    monkeypatch.setenv("VLM_MODEL", "test-vlm")
    main = load_app(monkeypatch, tmp_path, "test-vlm-vehicle-attributes")

    from app.services import vlm

    class FakeVLMResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return (
                b'{"choices":[{"message":{"content":"```json\\n{'
                b'\\"object_type\\":\\"vehicle\\",'
                b'\\"vehicle_color\\":\\"blue\\",'
                b'\\"vehicle_type\\":\\"suv\\",'
                b'\\"confidence\\":0.82}\\n```"}}]}'
            )

    monkeypatch.setattr(vlm.request, "urlopen", lambda *_args, **_kwargs: FakeVLMResponse())

    with TestClient(main.create_app()) as client:
        response = client.post(
            "/api/attributes/analyze",
            data={
                "object_type": "vehicle",
                "bbox_json": '{"x1":0,"y1":0,"x2":100,"y2":60,"label":"vehicle"}',
            },
            files={"file": ("vehicle.jpg", b"vehicle-image", "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    attributes = payload["items"][0]["attributes"]
    assert attributes["object_type"] == "vehicle"
    assert attributes["vehicle_color"] == "blue"
    assert attributes["vehicle_type"] == "suv"
    assert attributes["vehicle"] == {"color": "blue", "type": "suv"}


def test_visual_embedding_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "qwen3_vl")
    monkeypatch.setenv("VISUAL_EMBEDDING_MODEL", "Qwen3-VL-Embedding-2B")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2")
    monkeypatch.setenv("VISUAL_EMBEDDING_SERVICE_API_KEY", "secret-token")
    main = load_app(monkeypatch, tmp_path, "test-embedding-endpoint")

    from app.services.embeddings import VisualEmbeddingService

    monkeypatch.setattr(VisualEmbeddingService, "embed_text", lambda self, text: [0.1, 0.9])
    with TestClient(main.create_app()) as client:
        unauthorized_response = client.post(
            "/api/embeddings/visual",
            json={"text": "白色衣服"},
        )
        response = client.post(
            "/api/embeddings/visual",
            headers={"Authorization": "Bearer secret-token"},
            json={"text": "白色衣服"},
        )

    assert unauthorized_response.status_code == 401
    assert response.status_code == 200
    assert response.json() == {
        "embedding": [0.1, 0.9],
        "dim": 2,
        "model": "Qwen3-VL-Embedding-2B",
        "provider": "qwen3_vl",
    }


def test_visual_rerank_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_RERANK_PROVIDER", "qwen3_vl_reranker")
    monkeypatch.setenv("VLM_RERANK_MODEL", "Qwen3-VL-Reranker-2B-8bit")
    monkeypatch.setenv("VISUAL_EMBEDDING_SERVICE_API_KEY", "secret-token")
    main = load_app(monkeypatch, tmp_path, "test-rerank-endpoint")

    from app.services.rerank import RerankDecision, VisualRerankerService

    captured = {}

    def fake_rerank_image(self, query, image_path, attributes):
        captured["query"] = query
        captured["image_name"] = image_path.name
        captured["attributes"] = attributes
        return RerankDecision(score=0.87, matched=True, reason="reranker score")

    monkeypatch.setattr(VisualRerankerService, "rerank_image", fake_rerank_image)
    with TestClient(main.create_app()) as client:
        unauthorized_response = client.post(
            "/api/embeddings/rerank",
            json={"query": "光头的人", "image_base64": "aW1hZ2U="},
        )
        response = client.post(
            "/api/embeddings/rerank",
            headers={"Authorization": "Bearer secret-token"},
            json={
                "query": "光头的人",
                "image_base64": "aW1hZ2U=",
                "image_filename": "crop.jpg",
                "attributes": {"hair": "bald"},
            },
        )

    assert unauthorized_response.status_code == 401
    assert response.status_code == 200
    assert captured == {
        "query": "光头的人",
        "image_name": "query.jpg",
        "attributes": {"hair": "bald"},
    }
    assert response.json() == {
        "score": 0.87,
        "matched": True,
        "reason": "reranker score",
        "model": "Qwen3-VL-Reranker-2B-8bit",
        "provider": "qwen3_vl_reranker",
    }


def test_image_upload(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-upload")
    with TestClient(main.create_app()) as client:
        response = client.post(
            "/api/images/upload",
            files={"file": ("sample.jpg", b"fake-image", "image/jpeg")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["image_url"].startswith("/data/uploads/")

    with TestClient(main.create_app()) as client:
        get_response = client.get(f"/api/images/{payload['id']}")
        assert get_response.status_code == 200
        assert get_response.json()["id"] == payload["id"]

        process_response = client.post(f"/api/images/{payload['id']}/process")
        assert process_response.status_code == 200
        crops = process_response.json()
        assert len(crops) == 1
        assert crops[0]["image_id"] == payload["id"]
        assert crops[0]["crop_url"].startswith("/data/crops/")

        image_search_response = client.post(
            "/api/search/by-image",
            json={"image_id": payload["id"], "top_k": 5, "target": "person_crop", "filters": {}},
        )
        assert image_search_response.status_code == 200
        assert image_search_response.json() == {"items": []}


def test_visual_search_ignores_vector_and_uses_dataset_labels(monkeypatch, tmp_path):
    monkeypatch.setenv("MILVUS_ENABLED", "true")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "qwen3_vl")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2")
    main = load_app(monkeypatch, tmp_path, "test-vector-over-label-search")
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.search import VisualSearchService
    from app.services.vector_index import VectorSearchHit

    monkeypatch.setattr(
        VisualSearchService,
        "_try_vector_search",
        lambda self, object_type, payload: [VectorSearchHit(object_id=vector_crop_id, score=0.82)],
    )

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            label_image = Image(
                image_url="/data/datasets/phone_smoking/yolo/a.jpg",
                source_type="dataset",
            )
            vector_image = Image(image_url="/data/crops/vector.jpg", source_type="stream_frame")
            db.add_all([label_image, vector_image])
            db.commit()
            db.refresh(label_image)
            db.refresh(vector_image)
            db.add(
                PersonCrop(
                    image_id=label_image.id,
                    crop_url="/data/crops/smoking.jpg",
                    bbox={
                        "x": 1,
                        "y": 2,
                        "width": 30,
                        "height": 40,
                        "label": "smoking",
                        "dataset": "phone_smoking",
                    },
                )
            )
            vector_crop = PersonCrop(
                image_id=vector_image.id,
                crop_url="/data/crops/vector.jpg",
                bbox={"label": "person"},
            )
            db.add(vector_crop)
            db.commit()
            db.refresh(vector_crop)
            vector_crop_id = vector_crop.id

        response = client.post(
            "/api/search/person-crops",
            json={"query": "抽烟的人", "top_k": 5, "filters": {}},
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 1
    assert payload["items"][0]["crop_url"] == "/data/crops/smoking.jpg"
    assert payload["items"][0]["score"] == 1.0


def test_visual_search_uses_dataset_labels_when_vector_unavailable(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-label-search")
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            image = Image(
                image_url="/data/datasets/phone_smoking/yolo/a.jpg",
                source_type="dataset",
            )
            db.add(image)
            db.commit()
            db.refresh(image)
            db.add(
                PersonCrop(
                    image_id=image.id,
                    crop_url="/data/crops/smoking.jpg",
                    bbox={
                        "x": 1,
                        "y": 2,
                        "width": 30,
                        "height": 40,
                        "label": "smoking",
                        "dataset": "phone_smoking",
                    },
                )
            )
            db.commit()

        response = client.post(
            "/api/search/person-crops",
            json={"query": "抽烟的人", "top_k": 5, "filters": {}},
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 1
    assert payload["items"][0]["crop_url"] == "/data/crops/smoking.jpg"
    assert payload["items"][0]["score"] == 1.0


def test_observation_index_searches_by_person_name_and_returns_labels(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-observation-search")
    from app.db.session import SessionLocal
    from app.models.events import RecognitionEvent
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.services.observation_index import ObservationIndexService

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            person = Person(name="张三", employee_no="E001", department="研发部")
            image = Image(image_url="/data/frames/a.jpg", source_type="stream_frame")
            db.add_all([person, image])
            db.commit()
            db.refresh(person)
            db.refresh(image)
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/a.jpg",
                bbox={"label": "person"},
                person_id=person.id,
                captured_at=now,
                attributes={
                    "object_type": "person",
                    "clothing": {"upper_color": "black"},
                    "objects": {"backpack": True},
                    "has_backpack": True,
                    "top_color": "black",
                },
            )
            db.add(crop)
            db.commit()
            db.refresh(crop)
            db.add(
                RecognitionEvent(
                    image_id=image.id,
                    crop_id=crop.id,
                    person_id=person.id,
                    similarity=0.93,
                    result_type="known",
                    recognized_at=now,
                )
            )
            ObservationIndexService(db, main.get_settings()).upsert_crop(crop)
            db.commit()
            crop_id = crop.id
            person_id = person.id

        name_response = client.post(
            "/api/search/person-crops",
            json={"query": "张三", "top_k": 5, "filters": {}},
        )
        label_response = client.post(
            "/api/search/person-crops",
            json={"query": "黑衣背包的人", "top_k": 5, "filters": {}},
        )
        observation_response = client.get("/api/search/observations?query=张三&limit=10")

    assert name_response.status_code == 200
    name_payload = name_response.json()
    assert name_payload["items"][0]["crop_id"] == str(crop_id)
    assert name_payload["items"][0]["person_name"] == "张三"
    assert name_payload["items"][0]["person_id"] == str(person_id)
    assert name_payload["items"][0]["labels_zh"]["上衣颜色"] == "黑色"

    assert label_response.status_code == 200
    label_payload = label_response.json()
    assert label_payload["items"][0]["crop_id"] == str(crop_id)
    assert label_payload["items"][0]["person_name"] == "张三"

    assert observation_response.status_code == 200
    observation_payload = observation_response.json()
    assert observation_payload["total"] == 1
    assert observation_payload["items"][0]["crop_id"] == str(crop_id)
    assert observation_payload["items"][0]["person_name"] == "张三"
    assert observation_payload["items"][0]["has_face_embedding"] is False


def test_visual_search_does_not_run_vector_when_structured_matches(monkeypatch, tmp_path):
    monkeypatch.setenv("MILVUS_ENABLED", "true")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "qwen3_vl")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2")
    main = load_app(monkeypatch, tmp_path, "test-vector-with-structured-search")
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.search import VisualSearchService
    from app.services.vector_index import VectorSearchHit

    vector_calls = {"count": 0}

    def fake_vector_search(self, object_type, payload):
        vector_calls["count"] += 1
        return [VectorSearchHit(object_id=vector_crop_id, score=0.91)]

    monkeypatch.setattr(VisualSearchService, "_try_vector_search", fake_vector_search)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            structured_image = Image(image_url="/data/frames/structured.jpg", source_type="stream")
            vector_image = Image(image_url="/data/frames/vector.jpg", source_type="stream")
            db.add_all([structured_image, vector_image])
            db.commit()
            db.refresh(structured_image)
            db.refresh(vector_image)
            structured_crop = PersonCrop(
                image_id=structured_image.id,
                crop_url="/data/crops/structured.jpg",
                bbox={"label": "person"},
                attributes={
                    "object_type": "person",
                    "clothing": {"upper_color": "black"},
                    "objects": {"backpack": True},
                },
            )
            vector_crop = PersonCrop(
                image_id=vector_image.id,
                crop_url="/data/crops/vector.jpg",
                bbox={"label": "person"},
            )
            db.add_all([structured_crop, vector_crop])
            db.commit()
            db.refresh(vector_crop)
            vector_crop_id = vector_crop.id

        response = client.post(
            "/api/search/person-crops",
            json={"query": "黑衣背包的人", "top_k": 5, "filters": {}},
        )

    assert response.status_code == 200
    payload = response.json()
    assert vector_calls["count"] == 0
    assert [item["crop_url"] for item in payload["items"]] == [
        "/data/crops/structured.jpg"
    ]
    assert payload["items"][0]["score"] == 1.0


def test_label_search_does_not_fall_back_when_no_label_matches(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-label-search-fallback")
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.search import VisualSearchService
    from app.services.vector_index import VectorSearchHit

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            image = Image(image_url="/data/crops/person.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/person.jpg",
                bbox={"label": "person"},
            )
            db.add(crop)
            db.commit()
            db.refresh(crop)
            crop_id = crop.id

        monkeypatch.setattr(
            VisualSearchService,
            "_try_vector_search",
            lambda self, object_type, payload: [VectorSearchHit(object_id=crop_id, score=0.88)],
        )
        response = client.post(
            "/api/search/person-crops",
            json={"query": "看手机", "top_k": 5, "filters": {}},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"] == []


def test_visual_search_does_not_call_clip_vector_hits(monkeypatch, tmp_path):
    monkeypatch.setenv("MILVUS_ENABLED", "true")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "qwen3_vl")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2")
    monkeypatch.setenv("VISUAL_SEARCH_MIN_SCORE", "0.3")
    main = load_app(monkeypatch, tmp_path, "test-vector-min-score")
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.search import VisualSearchService
    from app.services.vector_index import VectorSearchHit

    crop_ids = {}

    def fake_vector_search(self, object_type, payload):
        return [
            VectorSearchHit(object_id=crop_ids["weak"], score=0.22),
            VectorSearchHit(object_id=crop_ids["strong"], score=0.41),
        ]

    monkeypatch.setattr(VisualSearchService, "_try_vector_search", fake_vector_search)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            image = Image(image_url="/data/crops/person.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            weak_crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/weak.jpg",
                bbox={"label": "person"},
            )
            strong_crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/strong.jpg",
                bbox={"label": "person"},
            )
            db.add_all([weak_crop, strong_crop])
            db.commit()
            db.refresh(weak_crop)
            db.refresh(strong_crop)
            crop_ids["weak"] = weak_crop.id
            crop_ids["strong"] = strong_crop.id

        response = client.post(
            "/api/search/person-crops",
            json={"query": "手机", "top_k": 5, "filters": {}},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"] == []


def test_visual_search_can_vlm_rerank_person_crop_candidates(monkeypatch, tmp_path):
    monkeypatch.setenv("MILVUS_ENABLED", "true")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "qwen3_vl")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2")
    monkeypatch.setenv("VLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("VLM_BASE_URL", "http://vlm.local/v1")
    monkeypatch.setenv("VLM_MODEL", "test-vlm")
    monkeypatch.setenv("VLM_RERANK_ENABLED", "true")
    monkeypatch.setenv("VLM_RERANK_CANDIDATE_LIMIT", "3")
    main = load_app(monkeypatch, tmp_path, "test-vlm-rerank")

    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.rerank import RerankDecision, VLMRerankService
    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    crop_ids = {}
    requested_top_k = {}

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)

    def fake_search_text(self, object_type, query, top_k):
        requested_top_k["value"] = top_k
        return [
            VectorSearchHit(object_id=crop_ids["weak"], score=0.92),
            VectorSearchHit(object_id=crop_ids["strong"], score=0.41),
            VectorSearchHit(object_id=crop_ids["middle"], score=0.6),
        ]

    def fake_rerank_image(self, query, image_path, attributes):
        score_by_name = {
            "weak.jpg": 0.12,
            "middle.jpg": 0.55,
            "strong.jpg": 0.96,
        }
        return RerankDecision(
            score=score_by_name[image_path.name],
            matched=image_path.name == "strong.jpg",
            reason=f"{image_path.stem} reason",
        )

    monkeypatch.setattr(MilvusVectorIndex, "search_text", fake_search_text)
    monkeypatch.setattr(VLMRerankService, "rerank_image", fake_rerank_image)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        data_dir = tmp_path / "data"
        (data_dir / "crops").mkdir(parents=True, exist_ok=True)
        for filename in ("weak.jpg", "middle.jpg", "strong.jpg"):
            (data_dir / "crops" / filename).write_bytes(filename.encode("utf-8"))
        with SessionLocal() as db:
            image = Image(image_url="/data/frames/source.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            for name in ("weak", "middle", "strong"):
                crop = PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/{name}.jpg",
                    bbox={"label": "person"},
                )
                db.add(crop)
                db.commit()
                db.refresh(crop)
                crop_ids[name] = crop.id

        response = client.post(
            "/api/search/person-crops",
            json={"query": "红衣戴帽的人", "top_k": 2, "filters": {}, "rerank": True},
        )

    assert response.status_code == 200
    assert requested_top_k == {}
    payload = response.json()
    assert payload["items"] == []


def test_visual_search_uses_configured_vlm_rerank_service_for_rerank_flag(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("MILVUS_ENABLED", "true")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "none")
    monkeypatch.setenv("VLM_RERANK_SERVICE_URL", "http://reranker.local")
    monkeypatch.setenv("VLM_RERANK_CANDIDATE_LIMIT", "3")
    main = load_app(monkeypatch, tmp_path, "test-vlm-rerank-service")

    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.rerank import RerankDecision, VLMRerankService
    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    crop_ids = {}
    requested_top_k = {}

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)

    def fake_search_text(self, object_type, query, top_k):
        requested_top_k["value"] = top_k
        return [
            VectorSearchHit(object_id=crop_ids["weak"], score=0.92),
            VectorSearchHit(object_id=crop_ids["strong"], score=0.41),
            VectorSearchHit(object_id=crop_ids["middle"], score=0.6),
        ]

    def fake_rerank_image(self, query, image_path, attributes):
        score_by_name = {
            "weak.jpg": 0.12,
            "middle.jpg": 0.55,
            "strong.jpg": 0.96,
        }
        return RerankDecision(
            score=score_by_name[image_path.name],
            matched=image_path.name == "strong.jpg",
            reason="reranker score",
        )

    monkeypatch.setattr(MilvusVectorIndex, "search_text", fake_search_text)
    monkeypatch.setattr(VLMRerankService, "rerank_image", fake_rerank_image)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        data_dir = tmp_path / "data"
        (data_dir / "crops").mkdir(parents=True, exist_ok=True)
        for filename in ("weak.jpg", "middle.jpg", "strong.jpg"):
            (data_dir / "crops" / filename).write_bytes(filename.encode("utf-8"))
        with SessionLocal() as db:
            image = Image(image_url="/data/frames/source.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            for name in ("weak", "middle", "strong"):
                crop = PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/{name}.jpg",
                    bbox={"label": "person"},
                )
                db.add(crop)
                db.commit()
                db.refresh(crop)
                crop_ids[name] = crop.id

        response = client.post(
            "/api/search/person-crops",
            json={"query": "红衣戴帽的人", "top_k": 2, "filters": {}, "rerank": True},
        )

    assert response.status_code == 200
    assert requested_top_k == {}
    payload = response.json()
    assert payload["items"] == []


def test_vlm_rerank_scores_candidates_concurrently(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("VLM_BASE_URL", "http://vlm.local/v1")
    monkeypatch.setenv("VLM_MODEL", "test-vlm")
    monkeypatch.setenv("VLM_RERANK_ENABLED", "true")
    monkeypatch.setenv("VLM_RERANK_CANDIDATE_LIMIT", "4")
    monkeypatch.setenv("VLM_RERANK_MAX_WORKERS", "4")
    load_app(monkeypatch, tmp_path, "test-vlm-rerank-parallel")

    import threading

    from app.config.settings import get_settings
    from app.schemas.media import SearchResultItem
    from app.services.rerank import RerankDecision, VLMRerankService

    crops_dir = tmp_path / "data" / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    names = ["a", "b", "c", "d"]
    for name in names:
        (crops_dir / f"{name}.jpg").write_bytes(name.encode("utf-8"))

    # Every candidate must be in flight at once for the barrier to clear; a serial loop
    # would deadlock here and trip the timeout instead.
    barrier = threading.Barrier(len(names), timeout=10)
    scores = {"a.jpg": 0.2, "b.jpg": 0.9, "c.jpg": 0.5, "d.jpg": 0.7}

    def fake_rerank_image(self, query, image_path, attributes):
        barrier.wait()
        return RerankDecision(
            score=scores[image_path.name],
            matched=True,
            reason=f"{image_path.stem} reason",
        )

    monkeypatch.setattr(VLMRerankService, "rerank_image", fake_rerank_image)

    items = [
        SearchResultItem(crop_url=f"/data/crops/{name}.jpg", score=0.1) for name in names
    ]
    reranked = VLMRerankService(get_settings()).rerank_person_crops(
        "红衣戴帽的人", items, limit=3
    )

    assert [item.crop_url for item in reranked] == [
        "/data/crops/b.jpg",
        "/data/crops/d.jpg",
        "/data/crops/c.jpg",
    ]
    assert [item.rerank_score for item in reranked] == [0.9, 0.7, 0.5]


def test_vlm_rerank_propagates_candidate_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("VLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("VLM_BASE_URL", "http://vlm.local/v1")
    monkeypatch.setenv("VLM_MODEL", "test-vlm")
    monkeypatch.setenv("VLM_RERANK_ENABLED", "true")
    load_app(monkeypatch, tmp_path, "test-vlm-rerank-failure")

    from app.config.settings import get_settings
    from app.schemas.media import SearchResultItem
    from app.services.rerank import VLMRerankService
    from app.services.vlm import VLMRuntimeError

    crops_dir = tmp_path / "data" / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for name in ("a", "b"):
        (crops_dir / f"{name}.jpg").write_bytes(name.encode("utf-8"))

    def fake_rerank_image(self, query, image_path, attributes):
        raise VLMRuntimeError("rerank backend is down")

    monkeypatch.setattr(VLMRerankService, "rerank_image", fake_rerank_image)

    items = [SearchResultItem(crop_url=f"/data/crops/{n}.jpg", score=0.1) for n in ("a", "b")]
    with pytest.raises(VLMRuntimeError):
        VLMRerankService(get_settings()).rerank_person_crops("查询", items, limit=2)


def test_visual_search_can_embedding_rerank_person_crop_candidates(monkeypatch, tmp_path):
    monkeypatch.setenv("MILVUS_ENABLED", "false")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "qwen3_vl")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2")
    monkeypatch.setenv("EMBEDDING_RERANK_ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_RERANK_CANDIDATE_LIMIT", "3")
    monkeypatch.setenv("VLM_PROVIDER", "none")
    main = load_app(monkeypatch, tmp_path, "test-embedding-rerank")

    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.embeddings import VisualEmbeddingService

    crop_ids = {}

    monkeypatch.setattr(VisualEmbeddingService, "is_enabled", lambda self: True)
    monkeypatch.setattr(VisualEmbeddingService, "embed_text", lambda self, text: [1.0, 0.0])

    def fake_embed_image(self, image_path):
        vectors = {
            "weak.jpg": [0.0, 1.0],
            "middle.jpg": [0.7, 0.7],
            "strong.jpg": [1.0, 0.0],
        }
        return vectors[image_path.name]

    monkeypatch.setattr(VisualEmbeddingService, "embed_image", fake_embed_image)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        data_dir = tmp_path / "data"
        (data_dir / "crops").mkdir(parents=True, exist_ok=True)
        for filename in ("weak.jpg", "middle.jpg", "strong.jpg"):
            (data_dir / "crops" / filename).write_bytes(filename.encode("utf-8"))
        with SessionLocal() as db:
            image = Image(image_url="/data/frames/source.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            for name in ("weak", "middle", "strong"):
                crop = PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/{name}.jpg",
                    bbox={"label": "person"},
                )
                db.add(crop)
                db.commit()
                db.refresh(crop)
                crop_ids[name] = crop.id

        response = client.post(
            "/api/search/person-crops",
            json={"query": "红衣戴帽的人", "top_k": 2, "filters": {}},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"] == []


def test_face_library_cache_matches_full_scan_and_reloads_after_enrollment(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-face-library-cache")

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.persons import Person
    from app.models.vectors import FaceEmbedding
    from app.services import faces as faces_module
    from app.services.faces import FaceRecognitionService

    def unit(*values: float) -> list[float]:
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]

    faces_module.invalidate_face_library_cache()
    settings = get_settings()
    with TestClient(main.create_app()), SessionLocal() as db:
        alice = Person(name="alice")
        bob = Person(name="bob")
        db.add_all([alice, bob])
        db.flush()
        db.add_all(
            [
                FaceEmbedding(person_id=alice.id, embedding=unit(1.0, 0.0, 0.0), face_model="m"),
                FaceEmbedding(person_id=bob.id, embedding=unit(0.0, 1.0, 0.0), face_model="m"),
                # A different dimension must stay out of a 3-d query's candidate set.
                FaceEmbedding(person_id=bob.id, embedding=unit(1.0, 1.0), face_model="m"),
            ]
        )
        db.commit()

        service = FaceRecognitionService(db, settings)
        query = unit(0.9, 0.1, 0.0)
        cached = service._search_matches(query, top_k=2)
        scanned = service._scan_matches(query, 2, None)

        assert [match.person.name for match in cached] == ["alice", "bob"]
        assert [match.face_embedding.id for match in cached] == [
            match.face_embedding.id for match in scanned
        ]
        assert cached[0].similarity == pytest.approx(scanned[0].similarity)

        library = service._face_library()
        assert sorted(library.buckets) == [2, 3]
        assert service._face_library() is library  # served from cache, not rebuilt

        carol = Person(name="carol")
        db.add(carol)
        db.flush()
        db.add(
            FaceEmbedding(person_id=carol.id, embedding=unit(0.95, 0.05, 0.0), face_model="m")
        )
        db.commit()
        faces_module.invalidate_face_library_cache()

        refreshed = service._search_matches(query, top_k=1)
        assert [match.person.name for match in refreshed] == ["carol"]

    faces_module.invalidate_face_library_cache()


def test_face_library_recognition_and_trajectory(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    def sample_face_bytes() -> bytes:
        image = np.full((96, 96, 3), 210, dtype=np.uint8)
        cv2.circle(image, (48, 44), 28, (160, 160, 160), -1)
        cv2.circle(image, (38, 38), 4, (20, 20, 20), -1)
        cv2.circle(image, (58, 38), 4, (20, 20, 20), -1)
        cv2.ellipse(image, (48, 52), (12, 7), 0, 0, 180, (40, 40, 40), 2)
        ok, buffer = cv2.imencode(".jpg", image)
        assert ok
        return buffer.tobytes()

    main = load_app(monkeypatch, tmp_path, "test-face-flow")
    content = sample_face_bytes()
    with TestClient(main.create_app()) as client:
        person_response = client.post(
            "/api/persons",
            json={"name": "王五", "employee_no": "E001"},
        )
        assert person_response.status_code == 200
        person = person_response.json()

        enroll_response = client.post(
            f"/api/persons/{person['id']}/faces",
            files={"file": ("face.jpg", content, "image/jpeg")},
        )
        assert enroll_response.status_code == 200
        face = enroll_response.json()
        assert face["person_id"] == person["id"]
        assert face["face_model"] == "opencv-gray32"

        faces_response = client.get(f"/api/persons/{person['id']}/faces")
        assert faces_response.status_code == 200
        assert len(faces_response.json()) == 1

        recognize_response = client.post(
            "/api/face/recognize?threshold=0.1",
            files={"file": ("query.jpg", content, "image/jpeg")},
        )
        assert recognize_response.status_code == 200
        recognition = recognize_response.json()
        assert recognition["result_type"] == "known"
        assert recognition["person"]["id"] == person["id"]
        assert recognition["similarity"] >= 0.99

        search_response = client.post(
            "/api/face/search?min_similarity=0.1",
            files={"file": ("query.jpg", content, "image/jpeg")},
        )
        assert search_response.status_code == 200
        assert search_response.json()["matches"][0]["person_id"] == person["id"]

        upload_response = client.post(
            "/api/images/upload",
            files={"file": ("frame.jpg", content, "image/jpeg")},
        )
        process_response = client.post(f"/api/images/{upload_response.json()['id']}/process")
        assert process_response.status_code == 200
        crops = process_response.json()
        assert crops[0]["person_id"] == person["id"]

        events_response = client.get(f"/api/persons/{person['id']}/events")
        assert events_response.status_code == 200
        assert len(events_response.json()) >= 2

        rebuild_response = client.post("/api/face/index/rebuild?limit=10")
        assert rebuild_response.status_code == 200
        assert rebuild_response.json()["seen"] >= 1

        trajectory_response = client.get(
            f"/api/persons/{person['id']}/trajectory?min_similarity=0.1"
        )
        assert trajectory_response.status_code == 200
        trajectory = trajectory_response.json()
        assert trajectory["person"]["id"] == person["id"]
        assert trajectory["items"][0]["person_id"] == person["id"]
        assert trajectory["items"][0]["similarity"] >= 0.99
        assert trajectory["items"][0]["face_bbox"]["width"] > 0


def test_face_enrollment_uses_insightface_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("FACE_EMBEDDING_PROVIDER", "insightface")
    monkeypatch.setenv("FACE_INSIGHTFACE_MODEL", "buffalo_l")
    # This test is about provider selection. The small-crop upscale pass is covered by
    # test_insightface_small_crop_extraction_upscales_before_embedding; switch it off here so
    # the 64x64 fixture does not pick up an "-upscaled-N.NNx" model suffix and a rescaled bbox.
    monkeypatch.setenv("FACE_CANDIDATE_UPSCALE_MIN_WIDTH", "0")
    monkeypatch.setenv("FACE_CANDIDATE_UPSCALE_MIN_HEIGHT", "0")
    main = load_app(monkeypatch, tmp_path, "test-face-insightface-provider")
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    class FakeFace:
        embedding = [1.0, 0.0, 0.0]
        bbox = [10.0, 12.0, 42.0, 50.0]
        det_score = 0.98

    class FakeFaceAnalysis:
        def __init__(self, name):
            self.name = name

        def prepare(self, ctx_id, det_size):
            assert ctx_id == 0
            assert det_size == (640, 640)

        def get(self, image):
            assert image is not None
            return [FakeFace()]

    insightface_module = types.ModuleType("insightface")
    insightface_app_module = types.ModuleType("insightface.app")
    insightface_app_module.FaceAnalysis = FakeFaceAnalysis
    monkeypatch.setitem(sys.modules, "insightface", insightface_module)
    monkeypatch.setitem(sys.modules, "insightface.app", insightface_app_module)

    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok

    with TestClient(main.create_app()) as client:
        person_response = client.post("/api/persons", json={"name": "Insight"})
        image_response = client.post(
            f"/api/persons/{person_response.json()['id']}/faces",
            files={"file": ("face.jpg", buffer.tobytes(), "image/jpeg")},
        )

    assert image_response.status_code == 200
    payload = image_response.json()
    assert payload["face_model"] == "insightface-buffalo_l"
    assert payload["quality_score"] == 0.98
    assert payload["face_bbox"]["width"] == 32


def test_insightface_cuda_algorithm_reports_buffalo_l_stack():
    from app.face_algorithms import InsightFaceCudaRecognizer

    info = InsightFaceCudaRecognizer(
        model_name="buffalo_l",
        device="cuda:0",
        root="/data/SightIndex/models/insightface",
    ).info()

    assert info.provider == "insightface"
    assert info.model_pack == "buffalo_l"
    assert info.detector == "RetinaFace-10GF"
    assert info.detector_model_file == "det_10g.onnx"
    assert info.recognizer == "ArcFace ResNet50@WebFace600K"
    assert info.recognizer_model_file == "w600k_r50.onnx"
    assert info.embedding_dim == 512
    assert info.model_root == "/data/SightIndex/models/insightface"
    assert info.ctx_id == 0
    assert info.requested_providers[0] == "CUDAExecutionProvider"


def test_insightface_small_crop_extraction_upscales_before_embedding(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    from app.config.settings import Settings
    from app.face_algorithms.insightface_cuda import FaceAlgorithmCandidate
    from app.services import faces as faces_module
    from app.services.faces import FaceRecognitionService

    image_path = tmp_path / "small-crop.jpg"
    image = np.full((80, 80, 3), 180, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    captured = {"det_size": None, "paths": []}

    class FakeRecognizer:
        def __init__(self, model_name, det_size, device, root, allow_download):
            captured["det_size"] = det_size
            captured["allow_download"] = allow_download

        def extract(self, path):
            captured["paths"].append(Path(path))
            loaded = cv2.imread(str(path))
            height, width = loaded.shape[:2]
            return [
                FaceAlgorithmCandidate(
                    embedding=[1.0, 0.0],
                    bbox={
                        "x": width / 4,
                        "y": height / 5,
                        "width": width / 4,
                        "height": height / 4,
                    },
                    quality_score=0.88,
                    model="fake-insightface",
                )
            ]

    monkeypatch.setattr(faces_module, "InsightFaceCudaRecognizer", FakeRecognizer)
    settings = Settings(
        data_dir=tmp_path / "data",
        face_embedding_provider="insightface",
        face_insightface_det_size=1280,
        face_candidate_upscale_min_width=240,
        face_candidate_upscale_min_height=240,
        face_candidate_upscale_max_factor=3.0,
    )
    service = FaceRecognitionService(db=None, settings=settings)

    candidates = service._extract_candidates(image_path, allow_fallback=False)

    assert captured["det_size"] == 1280
    assert len(captured["paths"]) == 2
    assert captured["paths"][0] == image_path
    assert captured["paths"][1].name.startswith("sightindex-face-upscale-")
    assert not captured["paths"][1].exists()
    assert candidates[0].model == "fake-insightface-upscaled-3.00x"
    assert candidates[0].bbox["x"] == pytest.approx(20.0)
    assert candidates[0].bbox["y"] == pytest.approx(16.0)
    assert candidates[0].bbox["width"] == pytest.approx(20.0)
    assert candidates[0].bbox["height"] == pytest.approx(20.0)


def test_face_enrollment_falls_back_when_insightface_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("FACE_EMBEDDING_PROVIDER", "insightface")
    main = load_app(monkeypatch, tmp_path, "test-face-insightface-fallback")
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok

    with TestClient(main.create_app()) as client:
        person_response = client.post("/api/persons", json={"name": "Fallback"})
        face_response = client.post(
            f"/api/persons/{person_response.json()['id']}/faces",
            files={"file": ("face.jpg", buffer.tobytes(), "image/jpeg")},
        )

    assert face_response.status_code == 200
    assert face_response.json()["face_model"] == "opencv-gray32"


def test_face_diagnostics_reports_recent_crop_match(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-face-diagnostics")

    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.models.vectors import FaceEmbedding
    from app.services.faces import FaceCandidate, FaceRecognitionService

    monkeypatch.setattr(
        FaceRecognitionService,
        "_best_candidate",
        lambda self, url, allow_fallback: FaceCandidate(
            embedding=[1.0, 0.0],
            bbox={"x": 4.0, "y": 5.0, "width": 60.0, "height": 64.0},
            quality_score=0.82,
            model="test",
        ),
    )

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            person = Person(name="诊断人员")
            image = Image(image_url="/data/frames/diagnostic.jpg", source_type="stream_frame")
            db.add_all([person, image])
            db.commit()
            db.refresh(person)
            db.refresh(image)
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/diagnostic.jpg",
                bbox={"label": "person"},
            )
            face = FaceEmbedding(
                person_id=person.id,
                embedding=[1.0, 0.0],
                face_model="test",
            )
            db.add_all([crop, face])
            db.commit()
            db.refresh(crop)
            crop_id = crop.id

        response = client.get("/api/face/diagnostics/recent?limit=5")

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["crop_id"] == str(crop_id)
    assert payload["items"][0]["verdict"] == "known"
    assert payload["items"][0]["top_person_name"] == "诊断人员"
    assert payload["items"][0]["top_similarity"] == 1.0
    assert payload["items"][0]["can_enroll"] is True


def test_enroll_person_face_from_crop(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-face-enroll-from-crop")

    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.services.faces import FaceCandidate, FaceRecognitionService

    monkeypatch.setattr(
        FaceRecognitionService,
        "_best_candidate",
        lambda self, url, allow_fallback: FaceCandidate(
            embedding=[0.0, 1.0],
            bbox={"x": 1.0, "y": 2.0, "width": 50.0, "height": 52.0},
            quality_score=0.91,
            model="test-crop",
        ),
    )

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            person = Person(name="补库人员")
            image = Image(image_url="/data/frames/from-crop.jpg", source_type="stream_frame")
            db.add_all([person, image])
            db.commit()
            db.refresh(person)
            db.refresh(image)
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/from-crop.jpg",
                bbox={"label": "person"},
            )
            db.add(crop)
            db.commit()
            db.refresh(crop)
            person_id = person.id
            crop_id = crop.id

        response = client.post(f"/api/persons/{person_id}/faces/from-crop/{crop_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["person_id"] == str(person_id)
    assert payload["crop_id"] == str(crop_id)
    assert payload["face_model"] == "test-crop"
    assert payload["face_bbox"]["width"] == 50.0


def test_person_trajectory_uses_vector_as_face_confirmation(monkeypatch, tmp_path):
    monkeypatch.setenv("MILVUS_ENABLED", "true")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "qwen3_vl")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2")
    monkeypatch.setenv("PERSON_TRAJECTORY_VECTOR_MIN_SCORE", "0.7")
    main = load_app(monkeypatch, tmp_path, "test-trajectory-vector")

    from app.db.session import SessionLocal
    from app.models.events import CountingEvent, RecognitionEvent
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.services.observation_index import ObservationIndexService
    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    crop_ids = {}

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)

    def fake_search_image(self, object_type, image_path, top_k):
        assert object_type == "person_crop"
        return [
            VectorSearchHit(object_id=crop_ids["seed"], score=1.0),
            VectorSearchHit(object_id=crop_ids["vector"], score=0.84),
        ]

    monkeypatch.setattr(MilvusVectorIndex, "search_image", fake_search_image)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        data_dir = tmp_path / "data"
        (data_dir / "crops").mkdir(parents=True, exist_ok=True)
        (data_dir / "crops" / "seed.jpg").write_bytes(b"seed")
        (data_dir / "crops" / "vector.jpg").write_bytes(b"vector")

        with SessionLocal() as db:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            person = Person(name="融合轨迹")
            seed_image = Image(image_url="/data/frames/seed.jpg", source_type="stream_frame")
            vector_image = Image(image_url="/data/frames/vector.jpg", source_type="stream_frame")
            db.add_all([person, seed_image, vector_image])
            db.commit()
            db.refresh(person)
            db.refresh(seed_image)
            db.refresh(vector_image)

            seed_crop = PersonCrop(
                image_id=seed_image.id,
                crop_url="/data/crops/seed.jpg",
                bbox={"x": 0, "y": 0, "width": 100, "height": 200},
                person_id=person.id,
                captured_at=now,
            )
            vector_crop = PersonCrop(
                image_id=vector_image.id,
                crop_url="/data/crops/vector.jpg",
                bbox={"x": 5, "y": 5, "width": 100, "height": 200},
                captured_at=now,
            )
            db.add_all([seed_crop, vector_crop])
            db.commit()
            db.refresh(seed_crop)
            db.refresh(vector_crop)
            crop_ids["seed"] = seed_crop.id
            crop_ids["vector"] = vector_crop.id

            db.add(
                RecognitionEvent(
                    image_id=seed_image.id,
                    crop_id=seed_crop.id,
                    person_id=person.id,
                    confidence=0.9,
                    similarity=0.95,
                    result_type="known",
                    recognized_at=now,
                )
            )
            ObservationIndexService(db, main.get_settings()).upsert_crop(seed_crop)
            db.add(
                CountingEvent(
                    image_id=vector_image.id,
                    crop_id=vector_crop.id,
                    count_type="line_crossing",
                    counted_at=now,
                )
            )
            db.commit()
            person_id = person.id

        response = client.get(f"/api/persons/{person_id}/trajectory")
        all_response = client.get(f"/api/persons/{person_id}/trajectory?mode=all")
        face_response = client.get(f"/api/persons/{person_id}/trajectory?mode=face")
        vector_response = client.get(f"/api/persons/{person_id}/trajectory?mode=vector")

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["crop_id"] == str(crop_ids["seed"])
    assert items[0]["match_source"] == "face"

    assert all_response.status_code == 200
    all_items = all_response.json()["items"]
    assert len(all_items) == 1
    assert all_items[0]["crop_id"] == str(crop_ids["seed"])
    assert all_items[0]["match_source"] == "face"

    items = all_items
    assert len(items) == 1
    assert items[0]["crop_id"] == str(crop_ids["seed"])
    assert items[0]["match_source"] == "face"

    assert face_response.status_code == 200
    face_items = face_response.json()["items"]
    assert len(face_items) == 1
    assert face_items[0]["crop_id"] == str(crop_ids["seed"])
    assert face_items[0]["match_source"] == "face"
    assert face_items[0]["vector_score"] is None

    assert vector_response.status_code == 200
    vector_items = vector_response.json()["items"]
    assert {item["crop_id"] for item in vector_items} == {
        str(crop_ids["seed"]),
        str(crop_ids["vector"]),
    }
    # VL-embedding matches are labelled by their true source, not a generic "vector".
    assert {item["match_source"] for item in vector_items} == {"vl_vector"}


def test_person_trajectory_releases_db_connection_before_vector_io(monkeypatch, tmp_path):
    monkeypatch.setenv("MILVUS_ENABLED", "true")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "qwen3_vl")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2")
    main = load_app(monkeypatch, tmp_path, "test-trajectory-vector-db-release")

    from app.db.session import SessionLocal
    from app.models.events import RecognitionEvent
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.services.vector_index import MilvusVectorIndex

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)

    def fake_search_image(self, object_type, image_path, top_k):
        with SessionLocal() as db:
            assert db.scalar(text("SELECT 1")) == 1
        return []

    monkeypatch.setattr(MilvusVectorIndex, "search_image", fake_search_image)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        data_dir = tmp_path / "data"
        (data_dir / "crops").mkdir(parents=True, exist_ok=True)
        (data_dir / "crops" / "seed.jpg").write_bytes(b"seed")

        with SessionLocal() as db:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            person = Person(name="连接释放")
            image = Image(image_url="/data/frames/seed.jpg", source_type="stream_frame")
            db.add_all([person, image])
            db.commit()
            db.refresh(person)
            db.refresh(image)
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/seed.jpg",
                bbox={"x": 0, "y": 0, "width": 100, "height": 200},
                person_id=person.id,
                captured_at=now,
            )
            db.add(crop)
            db.commit()
            db.refresh(crop)
            db.add(
                RecognitionEvent(
                    image_id=image.id,
                    crop_id=crop.id,
                    person_id=person.id,
                    confidence=0.9,
                    similarity=0.95,
                    result_type="known",
                    recognized_at=now,
                )
            )
            db.commit()
            person_id = person.id

        response = client.get(f"/api/persons/{person_id}/trajectory?mode=all")

    assert response.status_code == 200


def test_person_trajectory_uses_enrolled_face_image_as_vector_seed(monkeypatch, tmp_path):
    monkeypatch.setenv("MILVUS_ENABLED", "true")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "qwen3_vl")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2")
    monkeypatch.setenv("PERSON_TRAJECTORY_VECTOR_MIN_SCORE", "0.7")
    main = load_app(monkeypatch, tmp_path, "test-trajectory-face-seed")

    from app.db.session import SessionLocal
    from app.models.events import CountingEvent
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.models.vectors import FaceEmbedding
    from app.services.vector_index import MilvusVectorIndex, VectorSearchHit

    crop_ids = {}
    searched_paths = []

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)

    def fake_search_image(self, object_type, image_path, top_k):
        assert object_type == "person_crop"
        searched_paths.append(image_path.name)
        return [VectorSearchHit(object_id=crop_ids["vector"], score=0.42)]

    monkeypatch.setattr(MilvusVectorIndex, "search_image", fake_search_image)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        data_dir = tmp_path / "data"
        (data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (data_dir / "crops").mkdir(parents=True, exist_ok=True)
        (data_dir / "uploads" / "face.jpg").write_bytes(b"face")
        (data_dir / "crops" / "vector.jpg").write_bytes(b"vector")

        with SessionLocal() as db:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            person = Person(name="人脸种子")
            face_image = Image(image_url="/data/uploads/face.jpg", source_type="face_enrollment")
            vector_image = Image(image_url="/data/frames/vector.jpg", source_type="stream_frame")
            db.add_all([person, face_image, vector_image])
            db.commit()
            db.refresh(person)
            db.refresh(face_image)
            db.refresh(vector_image)

            vector_crop = PersonCrop(
                image_id=vector_image.id,
                crop_url="/data/crops/vector.jpg",
                bbox={"x": 5, "y": 5, "width": 100, "height": 200},
                captured_at=now,
            )
            face = FaceEmbedding(
                person_id=person.id,
                image_id=face_image.id,
                embedding=[1.0, 0.0],
                face_model="test",
            )
            db.add_all([vector_crop, face])
            db.commit()
            db.refresh(vector_crop)
            crop_ids["vector"] = vector_crop.id
            db.add(
                CountingEvent(
                    image_id=vector_image.id,
                    crop_id=vector_crop.id,
                    count_type="line_crossing",
                    counted_at=now,
                )
            )
            db.commit()
            person_id = person.id

        response = client.get(f"/api/persons/{person_id}/trajectory?mode=vector")

    assert response.status_code == 200
    items = response.json()["items"]
    assert searched_paths == ["face.jpg"]
    assert len(items) == 1
    assert items[0]["crop_id"] == str(crop_ids["vector"])
    assert items[0]["match_source"] == "vl_vector"
    assert items[0]["vector_score"] == 0.42


def test_person_trajectory_backfills_face_recognition_when_no_events(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-trajectory-face-backfill")

    from app.db.session import SessionLocal
    from app.models.events import RecognitionEvent
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.models.vectors import FaceEmbedding
    from app.services.faces import FaceRecognitionService

    calls = {"count": 0}
    person_id = None

    def fake_recognize_crop(self, crop, image, existing_event=None, require_ingest_enabled=True):
        assert require_ingest_enabled is False
        calls["count"] += 1
        event = existing_event or RecognitionEvent()
        event.image_id = image.id
        event.crop_id = crop.id
        event.person_id = person_id
        event.confidence = 0.8
        event.similarity = 0.93
        event.face_bbox = {"x": 1, "y": 2, "width": 20, "height": 20}
        event.result_type = "known"
        event.recognized_at = crop.captured_at or crop.created_at
        crop.person_id = person_id
        self.db.add(crop)
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    monkeypatch.setattr(FaceRecognitionService, "recognize_crop", fake_recognize_crop)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            person = Person(name="轨迹补识别")
            image = Image(image_url="/data/frames/backfill.jpg", source_type="stream_frame")
            db.add_all([person, image])
            db.commit()
            db.refresh(person)
            db.refresh(image)
            person_id = person.id
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/backfill.jpg",
                bbox={"label": "person"},
                captured_at=now,
                created_at=now,
            )
            face = FaceEmbedding(
                person_id=person.id,
                embedding=[1.0, 0.0],
                face_model="test",
            )
            db.add_all([crop, face])
            db.commit()
            db.refresh(crop)
            crop_id = crop.id

        vector_only_response = client.get(f"/api/persons/{person_id}/trajectory?mode=vector")
        assert calls["count"] == 0

        fast_response = client.get(f"/api/persons/{person_id}/trajectory")
        assert calls["count"] == 0

        response = client.get(
            f"/api/persons/{person_id}/trajectory?backfill_missing=true"
        )

    assert vector_only_response.status_code == 200
    assert vector_only_response.json()["items"] == []
    assert fast_response.status_code == 200
    assert fast_response.json()["items"] == []
    assert response.status_code == 200
    items = response.json()["items"]
    assert calls["count"] == 1
    assert len(items) == 1
    assert items[0]["crop_id"] == str(crop_id)
    assert items[0]["person_id"] == str(person_id)
    assert items[0]["match_source"] == "face"
    assert items[0]["similarity"] == 0.93
    assert items[0]["face_bbox"]["width"] == 20


def test_person_trajectory_does_not_repeat_current_unknown_backfill(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-trajectory-current-unknown-skip")

    from app.db.session import SessionLocal
    from app.models.events import RecognitionEvent
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.models.vectors import FaceEmbedding
    from app.services.faces import FaceRecognitionService

    calls = {"count": 0}
    person_id = None

    def fake_recognize_crop(self, crop, image, existing_event=None, require_ingest_enabled=True):
        calls["count"] += 1
        return existing_event

    monkeypatch.setattr(FaceRecognitionService, "recognize_crop", fake_recognize_crop)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            person = Person(name="已判未知")
            image = Image(image_url="/data/frames/unknown-skip.jpg", source_type="stream_frame")
            db.add_all([person, image])
            db.commit()
            db.refresh(person)
            db.refresh(image)
            person_id = person.id
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/unknown-skip.jpg",
                bbox={"label": "person"},
                captured_at=now,
                created_at=now,
            )
            face = FaceEmbedding(
                person_id=person.id,
                embedding=[1.0, 0.0],
                face_model="test",
            )
            db.add_all([crop, face])
            db.commit()
            db.refresh(crop)
            event = RecognitionEvent(
                image_id=image.id,
                crop_id=crop.id,
                result_type="unknown",
                recognized_at=now,
            )
            db.add(event)
            db.commit()

        response = client.get(f"/api/persons/{person_id}/trajectory?mode=face")

    assert response.status_code == 200
    assert response.json()["items"] == []
    assert calls["count"] == 0


def test_face_rebuild_updates_existing_unknown_crop_event(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-face-rebuild-updates-unknown")

    from app.db.session import SessionLocal
    from app.models.events import RecognitionEvent
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.models.vectors import FaceEmbedding
    from app.services.faces import FaceRecognitionService

    person_id = None
    original_event_id = None

    def fake_recognize_crop(self, crop, image, existing_event=None, require_ingest_enabled=True):
        assert require_ingest_enabled is False
        assert existing_event is not None
        existing_event.person_id = person_id
        existing_event.confidence = 0.86
        existing_event.similarity = 0.94
        existing_event.face_bbox = {"x": 1, "y": 2, "width": 18, "height": 18}
        existing_event.result_type = "known"
        crop.person_id = person_id
        self.db.add(crop)
        self.db.add(existing_event)
        self.db.commit()
        self.db.refresh(existing_event)
        return existing_event

    monkeypatch.setattr(FaceRecognitionService, "recognize_crop", fake_recognize_crop)

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            person = Person(name="未知重建")
            image = Image(image_url="/data/frames/rebuild.jpg", source_type="stream_frame")
            db.add_all([person, image])
            db.commit()
            db.refresh(person)
            db.refresh(image)
            person_id = person.id
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/rebuild.jpg",
                bbox={"label": "person"},
                created_at=now,
            )
            face = FaceEmbedding(
                person_id=person.id,
                embedding=[1.0, 0.0],
                face_model="test",
            )
            db.add_all([crop, face])
            db.commit()
            db.refresh(crop)
            event = RecognitionEvent(
                image_id=image.id,
                crop_id=crop.id,
                result_type="unknown",
                recognized_at=now,
            )
            db.add(event)
            db.commit()
            db.refresh(event)
            original_event_id = event.id

        response = client.post("/api/face/index/rebuild?limit=10")

        with SessionLocal() as db:
            updated_event = db.get(RecognitionEvent, original_event_id)

    assert response.status_code == 200
    payload = response.json()
    assert payload["events_created"] == 0
    assert payload["events_updated"] == 1
    assert payload["matched"] == 1
    assert updated_event is not None
    assert updated_event.person_id == person_id
    assert updated_event.result_type == "known"


def test_explicit_face_rebuild_marks_crop_without_face(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-face-rebuild-no-face-marker")

    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models.events import RecognitionEvent
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.models.vectors import FaceEmbedding
    from app.services.faces import FaceRecognitionService

    monkeypatch.setattr(
        FaceRecognitionService,
        "_best_candidate",
        lambda self, url, allow_fallback: None,
    )

    crop_id = None

    with TestClient(main.create_app()) as client:
        client.get("/health")
        with SessionLocal() as db:
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            person = Person(name="无脸标记")
            image = Image(image_url="/data/frames/no-face-marker.jpg", source_type="stream_frame")
            db.add_all([person, image])
            db.commit()
            db.refresh(person)
            db.refresh(image)
            crop = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/no-face-marker.jpg",
                bbox={"label": "person"},
                created_at=now,
            )
            face = FaceEmbedding(
                person_id=person.id,
                embedding=[1.0, 0.0],
                face_model="test",
            )
            db.add_all([crop, face])
            db.commit()
            db.refresh(crop)
            crop_id = crop.id

        response = client.post("/api/face/index/rebuild?limit=10")

        with SessionLocal() as db:
            event = db.scalar(select(RecognitionEvent).where(RecognitionEvent.crop_id == crop_id))

    assert response.status_code == 200
    payload = response.json()
    assert payload["events_created"] == 1
    assert payload["matched"] == 0
    assert event is not None
    assert event.result_type == "no_face"
    assert event.person_id is None


def test_visual_search_does_not_fallback_to_score_zero_when_vector_provider_fails(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MILVUS_ENABLED", "true")
    monkeypatch.setenv("VISUAL_EMBEDDING_PROVIDER", "qwen3_vl_http")
    monkeypatch.setenv("VISUAL_EMBEDDING_DIM", "2048")
    monkeypatch.delenv("VISUAL_EMBEDDING_SERVICE_URL", raising=False)
    main = load_app(monkeypatch, tmp_path, "test-vector-failure")
    with TestClient(main.create_app()) as client:
        upload_response = client.post(
            "/api/images/upload",
            files={"file": ("sample.jpg", b"fake-image", "image/jpeg")},
        )
        assert upload_response.status_code == 200

        process_response = client.post(f"/api/images/{upload_response.json()['id']}/process")
        assert process_response.status_code == 200
        assert len(process_response.json()) == 1

        search_response = client.post(
            "/api/search/person-crops",
            json={"query": "白色衣服", "top_k": 5, "filters": {}},
        )

    assert search_response.status_code == 200
    assert search_response.json() == {"items": []}


def test_video_upload_processes_frames(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5,
        (64, 48),
    )
    for index in range(4):
        frame = np.full((48, 64, 3), 40 + index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    main = load_app(monkeypatch, tmp_path, "test-video")
    with TestClient(main.create_app()) as client, video_path.open("rb") as video_file:
        response = client.post(
            "/api/videos/upload?frame_interval_seconds=0.2&max_frames=2",
            files={"file": ("sample.avi", video_file, "video/x-msvideo")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["video_url"].startswith("/data/videos/")
    assert payload["frames_sampled"] == 2
    assert payload["images_created"] == 2
    assert payload["crops_created"] == 2
    assert payload["counting_events_created"] == 0
    assert len(payload["image_ids"]) == 2
    assert len(payload["crop_ids"]) == 2

    with TestClient(main.create_app()) as client:
        images_response = client.get("/api/images?has_crops=true")
        counts_response = client.get("/api/media/counts")
        second_image_response = client.get("/api/images?has_crops=true&limit=1&offset=1")
        second_crop_response = client.get("/api/person-crops?limit=1&offset=1")
    assert images_response.status_code == 200
    images = images_response.json()
    assert len(images) == 2
    assert counts_response.status_code == 200
    assert counts_response.json() == {
        "image_with_crops_count": 2,
        "person_crop_count": 2,
    }
    assert second_image_response.status_code == 200
    assert len(second_image_response.json()) == 1
    assert second_image_response.json()[0]["id"] == images[1]["id"]
    assert second_crop_response.status_code == 200
    assert len(second_crop_response.json()) == 1
    assert all(image["thumbnail_url"].startswith("/data/thumbnails/") for image in images)
    for image in images:
        annotated_path = tmp_path / "data" / image["thumbnail_url"].removeprefix("/data/")
        assert annotated_path.exists()


def test_video_upload_queue_full_returns_503_with_exact_partial_progress(
    monkeypatch, tmp_path
):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    video_path = tmp_path / "sample-backpressure.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5,
        (64, 48),
    )
    for index in range(2):
        writer.write(np.full((48, 64, 3), 50 + index * 30, dtype=np.uint8))
    writer.release()

    main = load_app(monkeypatch, tmp_path, "test-video-backpressure")
    from app.services.frame_processing import FrameProcessingService
    from app.services.vector_index_queue import VectorQueueFullError

    original_process_image = FrameProcessingService.process_image
    attempts = 0

    def fail_second_frame(self, image, detections=None):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise VectorQueueFullError("target=person_crop limit=1")
        return original_process_image(self, image, detections=detections)

    monkeypatch.setattr(FrameProcessingService, "process_image", fail_second_frame)
    with TestClient(main.create_app()) as client, video_path.open("rb") as video_file:
        response = client.post(
            "/api/videos/upload?frame_interval_seconds=0.2&max_frames=2",
            files={"file": ("sample.avi", video_file, "video/x-msvideo")},
        )
        images = client.get("/api/images?has_crops=true").json()
        crops = client.get("/api/person-crops").json()

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "vector_index_queue_full"
    assert detail["partial"] is True
    assert detail["frames_sampled"] == 2
    assert detail["frames_processed"] == 1
    assert detail["images_committed"] == 1
    assert detail["crops_committed"] == 1
    assert detail["counting_events_committed"] == 0
    assert len(detail["image_ids"]) == 1
    assert len(detail["crop_ids"]) == 1
    assert len(images) == 1
    assert len(crops) == 1
    assert len(list((tmp_path / "data" / "crops").glob("*"))) == 1
    assert list((tmp_path / "data" / "videos").glob("*")) == []


def test_video_upload_with_counting_line_only_stores_crossing_frames(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    video_path = tmp_path / "sample-crossing.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5,
        (100, 100),
    )
    for index in range(4):
        frame = np.full((100, 100, 3), 40 + index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    main = load_app(monkeypatch, tmp_path, "test-video-crossing")

    from app.services.frame_processing import Detection, FrameProcessingService

    detections = [
        Detection(bbox={"x": 20, "y": 40, "width": 20, "height": 20}, confidence=1),
        Detection(bbox={"x": 30, "y": 40, "width": 20, "height": 20}, confidence=1),
        Detection(bbox={"x": 40, "y": 40, "width": 20, "height": 20}, confidence=1),
        Detection(bbox={"x": 50, "y": 40, "width": 20, "height": 20}, confidence=1),
    ]

    def fake_detect(self, image_path):
        return [detections.pop(0)]

    monkeypatch.setattr(FrameProcessingService, "detect_image_path", fake_detect)
    with TestClient(main.create_app()) as client, video_path.open("rb") as video_file:
        response = client.post(
            "/api/videos/upload"
            "?frame_interval_seconds=0.2&max_frames=4"
            "&line_x1=0.5&line_y1=0&line_x2=0.5&line_y2=1",
            files={"file": ("sample-crossing.avi", video_file, "video/x-msvideo")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["frames_sampled"] == 4
    assert payload["images_created"] == 1
    assert payload["crops_created"] == 1
    assert payload["counting_events_created"] == 1
    assert len(payload["image_ids"]) == 1
    assert len(payload["crop_ids"]) == 1


def test_line_crossing_counter(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    main = load_app(monkeypatch, tmp_path, "test-line-counter")
    with TestClient(main.create_app()):
        from sqlalchemy import func, select

        from app.db.session import SessionLocal
        from app.models.events import CountingEvent
        from app.services.frame_processing import Detection
        from app.services.video_processing import CountingLine, VideoProcessingService

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        line = CountingLine(x1=0.5, y1=0.0, x2=0.5, y2=1.0)
        with SessionLocal() as db:
            service = VideoProcessingService(db, main.get_settings())
            tracks = {}
            next_track_id = 1
            count, next_track_id = service._count_line_crossings(
                detections=[
                    Detection(bbox={"x": 40, "y": 40, "width": 10, "height": 10}, confidence=1)
                ],
                frame=frame,
                line=line,
                tracks=tracks,
                next_track_id=next_track_id,
                counted_at=datetime.now(),
                camera_id=None,
                location_id=None,
            )
            assert count == 0
            count, next_track_id = service._count_line_crossings(
                detections=[
                    Detection(bbox={"x": 50, "y": 40, "width": 10, "height": 10}, confidence=1)
                ],
                frame=frame,
                line=line,
                tracks=tracks,
                next_track_id=next_track_id,
                counted_at=datetime.now(),
                camera_id=None,
                location_id=None,
                stream_id=None,
            )
            assert count == 1
            db.commit()
            total = db.scalar(select(func.count()).select_from(CountingEvent))
            assert total == 1


def test_line_crossing_uses_bottom_center_by_default(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-line-bottom-center")
    with TestClient(main.create_app()):
        from app.services.frame_processing import Detection
        from app.services.video_processing import CountingLine, VideoProcessingService

        line = CountingLine(x1=0.0, y1=0.5, x2=1.0, y2=0.5)

        from app.db.session import SessionLocal

        with SessionLocal() as db:
            service = VideoProcessingService(db, main.get_settings())
            bottom_point = service._detection_center(
                Detection(bbox={"x": 40, "y": 20, "width": 20, "height": 40}, confidence=1),
                frame_width=100,
                frame_height=100,
            )
            assert bottom_point == (0.5, 0.6)
            assert service._line_side(line, bottom_point) > 0


def test_line_crossing_matches_faster_motion(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    main = load_app(monkeypatch, tmp_path, "test-line-fast-motion")
    with TestClient(main.create_app()):
        from app.db.session import SessionLocal
        from app.services.frame_processing import Detection
        from app.services.video_processing import CountingLine, VideoProcessingService

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        line = CountingLine(x1=0.5, y1=0.0, x2=0.5, y2=1.0)
        with SessionLocal() as db:
            service = VideoProcessingService(db, main.get_settings())
            tracks = {}
            next_track_id = 1
            count, next_track_id = service._count_line_crossings(
                detections=[
                    Detection(bbox={"x": 20, "y": 20, "width": 20, "height": 60}, confidence=1)
                ],
                frame=frame,
                line=line,
                tracks=tracks,
                next_track_id=next_track_id,
                counted_at=datetime.now(),
                camera_id=None,
                location_id=None,
            )
            assert count == 0
            count, next_track_id = service._count_line_crossings(
                detections=[
                    Detection(bbox={"x": 50, "y": 20, "width": 20, "height": 60}, confidence=1)
                ],
                frame=frame,
                line=line,
                tracks=tracks,
                next_track_id=next_track_id,
                counted_at=datetime.now(),
                camera_id=None,
                location_id=None,
            )
            assert count == 1


def test_rtsp_capture_configures_tcp(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-rtsp-capture")
    from app.services.opencv_capture import open_video_capture

    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)

    class FakeCapture:
        def __init__(self):
            self.properties = []

        def isOpened(self):
            return True

        def release(self):
            return None

        def set(self, key, value):
            self.properties.append((key, value))
            return True

    class FakeCV2:
        CAP_FFMPEG = 1900
        CAP_PROP_OPEN_TIMEOUT_MSEC = 53
        CAP_PROP_READ_TIMEOUT_MSEC = 54
        CAP_PROP_BUFFERSIZE = 38
        calls = []

        @classmethod
        def VideoCapture(cls, *args):
            cls.calls.append(args)
            return FakeCapture()

    open_video_capture(FakeCV2, "rtsp://example.invalid/live", main.get_settings())

    assert "rtsp_transport;tcp" in os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
    assert FakeCV2.calls[0][0] == "rtsp://example.invalid/live"
    assert FakeCV2.calls[0][1] == FakeCV2.CAP_FFMPEG
    assert FakeCV2.calls[0][2] == [
        FakeCV2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        8000,
        FakeCV2.CAP_PROP_READ_TIMEOUT_MSEC,
        8000,
    ]


def test_stream_runtime_rejects_large_frame_jump(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    main = load_app(monkeypatch, tmp_path, "test-stream-frame-quality")
    from app.services.stream_runtime import StreamRuntime

    settings = main.get_settings().model_copy(
        update={"stream_corrupt_frame_mean_diff_threshold": 20.0}
    )
    runtime = StreamRuntime()
    baseline = np.full((120, 160, 3), 50, dtype=np.uint8)
    small_change = np.full((120, 160, 3), 55, dtype=np.uint8)
    corrupt_jump = np.full((120, 160, 3), 180, dtype=np.uint8)

    reference = runtime._frame_reference(baseline, cv2)

    assert runtime._usable_frame_reference(small_change, reference, cv2, settings)[0]
    assert not runtime._usable_frame_reference(corrupt_jump, reference, cv2, settings)[0]


def test_stream_registration(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-stream")
    with TestClient(main.create_app()) as client:
        response = client.post(
            "/api/streams",
            json={
                "name": "一楼门口",
                "stream_url": "rtsp://example.invalid/live",
                "protocol": "rtsp",
                "frame_interval_seconds": 2,
                "counting_line": {"x1": 0.25, "y1": 0.1, "x2": 0.25, "y2": 0.9},
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["name"] == "一楼门口"
        assert payload["status"] == "stopped"
        assert payload["counting_line"] == {"x1": 0.25, "y1": 0.1, "x2": 0.25, "y2": 0.9}

        list_response = client.get("/api/streams")
        assert list_response.status_code == 200
        assert len(list_response.json()) == 1
        assert list_response.json()[0]["counting_line"]["y2"] == 0.9

        update_response = client.patch(
            f"/api/streams/{payload['id']}/counting-line",
            json={"counting_line": {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.2}},
        )
        assert update_response.status_code == 200
        assert update_response.json()["counting_line"] == {
            "x1": 0.1,
            "y1": 0.2,
            "x2": 0.8,
            "y2": 0.2,
        }

        clear_response = client.patch(
            f"/api/streams/{payload['id']}/counting-line",
            json={"counting_line": None},
        )
        assert clear_response.status_code == 200
        assert clear_response.json()["counting_line"] is None

        from app.models.media import Image
        from app.services.media import MediaService

        def fake_snapshot(self, stream):
            image = Image(image_url="/data/frames/snapshot.jpg", source_type="stream_frame")
            self.db.add(image)
            self.db.flush()
            stream.last_frame_image_id = image.id
            self.db.add(stream)
            self.db.commit()
            self.db.refresh(image)
            return image

        monkeypatch.setattr(MediaService, "capture_stream_snapshot", fake_snapshot)
        snapshot_response = client.post(f"/api/streams/{payload['id']}/snapshot")
        assert snapshot_response.status_code == 200
        assert snapshot_response.json()["image_url"] == "/data/frames/snapshot.jpg"

        def fake_mjpeg(self, stream_url, fps=6.0, jpeg_quality=80):
            assert stream_url == "rtsp://example.invalid/live"
            assert fps == 1.0
            assert jpeg_quality == 70
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nfake-frame\r\n"

        monkeypatch.setattr(MediaService, "stream_mjpeg_frames", fake_mjpeg)
        mjpeg_response = client.get(f"/api/streams/{payload['id']}/mjpeg?fps=1&jpeg_quality=70")
        assert mjpeg_response.status_code == 200
        assert mjpeg_response.headers["content-type"].startswith("multipart/x-mixed-replace")
        assert b"fake-frame" in mjpeg_response.content

        counts_response = client.get(f"/api/streams/{payload['id']}/counts")
        assert counts_response.status_code == 200
        assert counts_response.json()["counting_event_count"] == 0

        from app.db.session import SessionLocal
        from app.models.events import CountingEvent

        with SessionLocal() as db:
            count_event = CountingEvent(
                stream_id=uuid.UUID(payload["id"]),
                counted_at=datetime.now(ZoneInfo("Asia/Shanghai")),
            )
            db.add(count_event)
            db.commit()
            event_id = count_event.id

        delete_response = client.delete(f"/api/streams/{payload['id']}")
        assert delete_response.status_code == 200
        assert delete_response.json()["status"] == "deleted"

        deleted_response = client.get(f"/api/streams/{payload['id']}")
        assert deleted_response.status_code == 404

        list_after_delete_response = client.get("/api/streams")
        assert list_after_delete_response.status_code == 200
        assert list_after_delete_response.json() == []

        with SessionLocal() as db:
            assert db.get(CountingEvent, event_id).stream_id is None

        invalid_response = client.post(
            "/api/streams",
            json={
                "name": "无效计数线",
                "stream_url": "rtsp://example.invalid/live",
                "counting_line": {"x1": 0.5, "y1": 0.5, "x2": 0.5, "y2": 0.5},
            },
        )
        assert invalid_response.status_code == 400


def test_stream_runtime_indexes_created_frame_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_INDEX_ON_INGEST", "true")
    monkeypatch.setenv("VECTOR_INDEX_ON_INGEST_BACKGROUND", "false")
    main = load_app(monkeypatch, tmp_path, "test-stream-runtime-index")
    from app.db.session import SessionLocal
    from app.models.media import VideoStream
    from app.services.stream_runtime import StreamRuntime

    indexed = []

    class FakeIndex:
        def flush(self, target):
            assert target == "image"

    class FakeVectorIndexingService:
        def __init__(self, db, settings):
            self.db = db
            self.settings = settings
            self.index = FakeIndex()

        def write_image_vector(self, image, flush=False):
            assert flush is False
            indexed.append(image.id)

        def record_image_index(self, image):
            return None

    monkeypatch.setattr(
        "app.services.vector_index.VectorIndexingService",
        FakeVectorIndexingService,
    )
    runtime = StreamRuntime()

    main.init_db()
    with SessionLocal() as db:
        stream = VideoStream(name="入口", stream_url="rtsp://example.invalid/live")
        db.add(stream)
        db.commit()
        db.refresh(stream)

        image = runtime._create_frame_image(
            db,
            stream,
            "/data/frames/runtime-frame.jpg",
            datetime.now(ZoneInfo("Asia/Shanghai")),
        )
        db.commit()
        image_id = image.id
        runtime._try_index_frame_image(db, image, main.get_settings())

    assert indexed == [image_id]


def test_stream_runtime_rolls_back_queue_full_and_processes_next_frame(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("STREAM_WARMUP_FRAMES", "0")
    main = load_app(monkeypatch, tmp_path, "test-stream-queue-backpressure")
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop, VideoStream
    from app.models.vectors import VectorIndexJob
    from app.services.frame_processing import Detection, FrameProcessingService
    from app.services.stream_runtime import StreamRuntime
    from app.services.vector_index_queue import VectorQueueFullError

    class FakeFrame:
        # Real frames are numpy arrays; visit de-duplication reads the shape to normalise
        # detection centres, so a bare object() no longer stands in for one.
        shape = (720, 1280, 3)

    class FakeCapture:
        def __init__(self):
            self.frames = 0
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            self.frames += 1
            return True, FakeFrame()

        def release(self):
            self.released = True

    capture = FakeCapture()
    runtime = StreamRuntime()
    stop_event = threading.Event()
    errors = []
    written_frames = 0
    attempts = 0

    monkeypatch.setattr(
        "app.services.stream_runtime.open_video_capture",
        lambda *args: capture,
    )
    monkeypatch.setattr(
        runtime,
        "_usable_frame_reference",
        lambda *args: (True, None),
    )

    def write_frame(stream, frame, cv2):
        nonlocal written_frames
        written_frames += 1
        path = tmp_path / f"stream-{written_frames}.jpg"
        path.write_bytes(b"frame")
        return f"/data/frames/{path.name}", path, datetime.now()

    monkeypatch.setattr(runtime, "_write_frame_file", write_frame)
    # A detection needs a bbox now: visit de-duplication reads it to decide whether this body
    # was already stored earlier in the same visit.
    fake_detection = Detection(
        bbox={"x": 10, "y": 20, "width": 60, "height": 140}, confidence=0.9
    )
    monkeypatch.setattr(
        FrameProcessingService,
        "detect_image_path",
        lambda *args: [fake_detection],
    )
    monkeypatch.setattr(
        FrameProcessingService,
        "quality_filter_detections",
        lambda self, detections: detections,
    )

    def process_frame(self, image, detections=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            self.db.add(
                PersonCrop(
                    image_id=image.id,
                    crop_url="/data/crops/rejected.jpg",
                    bbox={},
                )
            )
            self.db.flush()
            raise VectorQueueFullError("target=person_crop limit=1")
        stop_event.set()
        return []

    monkeypatch.setattr(FrameProcessingService, "process_image", process_frame)
    monkeypatch.setattr(runtime, "_queue_backoff_seconds", lambda **kwargs: 0.0)
    monkeypatch.setattr("app.services.stream_runtime.time.sleep", lambda seconds: None)
    original_set_stream_error = runtime._set_stream_error

    def record_stream_error(db, stream, message, status="error"):
        errors.append((message, status))
        original_set_stream_error(db, stream, message, status)

    monkeypatch.setattr(runtime, "_set_stream_error", record_stream_error)

    main.init_db()
    with SessionLocal() as db:
        stream = VideoStream(
            name="queue-test",
            stream_url="rtsp://example.invalid/live",
            frame_interval_seconds=0.2,
            status="running",
        )
        db.add(stream)
        db.commit()
        stream_id = stream.id

    runtime._run_capture_loop(stream_id, stop_event)

    with SessionLocal() as db:
        persisted_crops = db.query(PersonCrop).all()
        persisted_images = db.query(Image).all()
        persisted_jobs = db.query(VectorIndexJob).all()
        persisted_stream = db.get(VideoStream, stream_id)

    assert attempts == 2
    assert capture.released is True
    assert persisted_crops == []
    assert len(persisted_images) == 1
    assert persisted_images[0].image_url.endswith("stream-2.jpg")
    assert persisted_jobs == []
    assert errors and "queue full" in errors[0][0].lower()
    assert errors[0][1] == "running"
    assert persisted_stream is not None
    assert persisted_stream.status == "stopped"
    assert not (tmp_path / "stream-1.jpg").exists()
    assert (tmp_path / "stream-2.jpg").exists()


def test_stream_snapshot_indexes_image_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTOR_INDEX_ON_INGEST", "true")
    monkeypatch.setenv("VECTOR_INDEX_ON_INGEST_BACKGROUND", "false")
    main = load_app(monkeypatch, tmp_path, "test-stream-snapshot-index")
    from app.db.session import SessionLocal
    from app.models.media import VideoStream
    from app.services.media import MediaService

    cv2 = types.SimpleNamespace(
        IMWRITE_JPEG_QUALITY=1,
        imwrite=lambda path, frame, params: True,
    )

    class FakeCapture:
        def isOpened(self):
            return True

        def read(self):
            return True, object()

        def release(self):
            return None

    indexed = []

    def fake_index_image(self, image):
        indexed.append(image.id)

    monkeypatch.setattr("app.services.media.open_video_capture", lambda *args: FakeCapture())
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(MediaService, "_try_index_image", fake_index_image)

    main.init_db()
    with SessionLocal() as db:
        stream = VideoStream(name="入口", stream_url="rtsp://example.invalid/live")
        db.add(stream)
        db.commit()
        db.refresh(stream)

        image = MediaService(db, main.get_settings()).capture_stream_snapshot(stream)

    assert indexed == [image.id]


def test_running_streams_autostart_on_app_lifespan(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-stream-autostart")
    from app.db.session import SessionLocal
    from app.models.media import VideoStream
    from app.services.stream_runtime import stream_runtime

    started = []

    def fake_start(stream_id):
        started.append(stream_id)
        return "started"

    monkeypatch.setattr(stream_runtime, "start", fake_start)
    main.init_db()

    with SessionLocal() as db:
        running_stream = VideoStream(
            name="running",
            stream_url="rtsp://example.invalid/running",
            protocol="rtsp",
            status="running",
        )
        stopped_stream = VideoStream(
            name="stopped",
            stream_url="rtsp://example.invalid/stopped",
            protocol="rtsp",
            status="stopped",
        )
        db.add_all([running_stream, stopped_stream])
        db.commit()
        running_id = running_stream.id
        stopped_id = stopped_stream.id

    with TestClient(main.create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert started == [running_id]
    assert stopped_id not in started
