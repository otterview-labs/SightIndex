import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_reid import load_app


@pytest.fixture
def demo(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch, tmp_path, "public-demo",
        PUBLIC_DEMO_READ_ONLY="true", PERSON_DETECTOR="hog",
        PERSON_CROP_MIN_CONFIDENCE="0", PUBLIC_BASE_URL="http://testserver",
        FACE_RECOGNITION_ON_INGEST="false", FACE_INSIGHTFACE_ALLOW_DOWNLOAD="false",
        STREAM_AUTOSTART_RUNNING="false", MILVUS_ENABLED="false", REID_ENABLED="false",
        VECTOR_INDEX_ON_INGEST="false", VLM_STRUCTURED_ON_INGEST="false",
        VLM_STRUCTURED_BACKGROUND="false", EMBEDDING_PROVIDER="none",
        VISUAL_EMBEDDING_PROVIDER="none", VLM_PROVIDER="none",
        APP_BASIC_AUTH_USERNAME="", APP_BASIC_AUTH_PASSWORD="",
    )
    from app.public_demo_seed import seed

    seed()
    return main


def test_real_sample_search_and_read_only_boundary(demo):
    with TestClient(demo.create_app()) as client:
        before = client.get("/api/media/counts").json()
        assert before["image_with_crops_count"] == 1
        assert before["person_crop_count"] > 0
        result = client.post("/api/search/person-crops", json={"query": "深色"})
        assert result.status_code == 200
        assert result.json()["items"]
        crop = result.json()["items"][0]
        assert client.get(crop["crop_url"]).status_code == 200
        assert client.get("/data/uploads/nasa-walk.jpg").status_code == 200
        assert client.get("/api/search/observations").json()["total"] == before["person_crop_count"]
        empty = client.post("/api/search/person-crops", json={"query": "浅色上衣"})
        assert empty.json() == {"items": []}
        blocked = [
            ("POST", "/api/images/upload"), ("POST", "/api/streams"),
            ("POST", f"/api/images/{crop['image_id']}/process"),
            ("DELETE", f"/api/images/{crop['image_id']}"),
            ("PATCH", f"/api/images/{crop['image_id']}/tags"),
            ("POST", "/api/face/diagnostics/crops"), ("POST", "/api/reid/search"),
            ("POST", "/api/vlm/chat/completions"), ("POST", "/v1/embeddings"),
            ("POST", "/api/chat"), ("GET", "/api/new-future-endpoint"),
            ("GET", "/docs"), ("GET", "/openapi.json"),
        ]
        for method, path in blocked:
            assert client.request(method, path).status_code == 403, path
        assert client.get("/api/media/counts").json() == before
        assert client.get("/api/streams").json() == []
        assert client.post("/api/search/person-crops", content=b"x" * 4097).status_code == 413


def test_seed_refuses_an_existing_database(demo):
    from app.public_demo_seed import seed

    with pytest.raises(RuntimeError, match="already exists"):
        seed()


@pytest.mark.parametrize("setting,value", [
    ("milvus_enabled", True), ("stream_autostart_running", True),
    ("face_recognition_on_ingest", True), ("vlm_provider", "openai"),
    ("database_url", "postgresql://example/demo"),
])
def test_demo_rejects_unsafe_configuration(demo, setting, value):
    from app.public_demo import validate_public_demo

    settings = demo.get_settings().model_copy(update={setting: value})
    with pytest.raises(ValueError):
        validate_public_demo(settings)


def test_chunked_body_is_bounded_before_handler():
    from app.public_demo import PublicDemoGuard

    called = False

    async def handler(scope, receive, send):
        nonlocal called
        called = True

    messages = iter([
        {"type": "http.request", "body": b"x" * 2048, "more_body": True},
        {"type": "http.request", "body": b"x" * 2049, "more_body": False},
    ])
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    asyncio.run(PublicDemoGuard(handler)(
        {"type": "http", "method": "POST", "path": "/api/search/person-crops"}, receive, send,
    ))
    assert not called
    assert sent[0]["status"] == 413
    assert "4 KB" in json.loads(sent[1]["body"])["detail"]


def test_budget_is_bounded_and_health_remains_available():
    from app.public_demo import PublicDemoGuard

    app = FastAPI()

    @app.get("/health")
    @app.get("/api/media/counts")
    def okay():
        return {"status": "ok"}

    app.add_middleware(PublicDemoGuard, requests_per_minute=2)
    with TestClient(app) as client:
        for _ in range(2):
            assert client.get("/api/media/counts").status_code == 200
        response = client.get("/api/media/counts")
        assert response.status_code == 429
        assert response.headers["retry-after"] == "60"
        assert client.get("/health").status_code == 200
