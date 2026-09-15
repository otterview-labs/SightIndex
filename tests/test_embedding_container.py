import base64
import importlib
import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image
from test_reid import load_app


@pytest.fixture
def embedding_module(monkeypatch, tmp_path):
    load_app(
        monkeypatch, tmp_path, "embedding-container",
        VISUAL_EMBEDDING_SERVICE_API_KEY="unit-test-key",
        VISUAL_EMBEDDING_MODEL="/unused",
        MILVUS_ENABLED="false",
        REID_ENABLED="false",
        VECTOR_INDEX_ON_INGEST="false",
    )
    sys.modules.pop("deploy.containers.embedding_app", None)
    module = importlib.import_module("deploy.containers.embedding_app")

    class Vector:
        def tolist(self):
            return [1.0] + [0.0] * (module.EMBEDDING_DIM - 1)

    class Model:
        def process(self, items):
            return [Vector() for _item in items]

    model = Model()
    monkeypatch.setattr(module, "load_model", lambda _path: model)
    return module, model


def test_embedding_requires_auth_and_warmup_before_readiness(embedding_module):
    module, _model = embedding_module
    app = module.create_app()
    assert TestClient(app).get("/health").status_code == 503
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["ready"] is True
        assert health.json()["dim"] == 2048
        assert client.post("/api/embeddings/visual", json={"text": "hello"}).status_code == 401
        response = client.post(
            "/api/embeddings/visual",
            headers={"Authorization": "Bearer unit-test-key"},
            json={"text": "红色衣服"},
        )
        assert response.status_code == 200
        assert response.json()["dim"] == 2048
        assert response.json()["model"] == "Qwen/Qwen3-VL-Embedding-2B"
        assert client.post("/api/persons").status_code == 401
    assert app.state.ready is False


def test_embedding_accepts_images_but_rejects_bad_content(embedding_module):
    module, _model = embedding_module
    buffer = io.BytesIO()
    Image.new("RGB", (32, 64), (96, 96, 96)).save(buffer, format="PNG")
    with TestClient(module.create_app()) as client:
        headers = {"X-API-Key": "unit-test-key"}
        response = client.post(
            "/api/embeddings/visual", headers=headers,
            json={"image_base64": base64.b64encode(buffer.getvalue()).decode()},
        )
        assert response.status_code == 200
        invalid = client.post(
            "/api/embeddings/visual", headers=headers,
            json={"image_base64": base64.b64encode(b"not an image").decode()},
        )
        assert invalid.status_code == 400
        assert client.post(
            "/api/embeddings/visual", headers=headers, json={"text": "x" * 8193}
        ).status_code == 422


def test_embedding_rejects_parallel_gpu_work_and_recovers(embedding_module, monkeypatch):
    module, model = embedding_module
    started = threading.Event()
    release = threading.Event()
    process = model.process

    with TestClient(module.create_app()) as client:
        def blocking(items):
            started.set()
            assert release.wait(timeout=5)
            return process(items)

        monkeypatch.setattr(model, "process", blocking)
        headers = {"X-API-Key": "unit-test-key"}
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                client.post, "/api/embeddings/visual", headers=headers, json={"text": "first"}
            )
            try:
                assert started.wait(timeout=5)
                assert client.post(
                    "/api/embeddings/visual", headers=headers, json={"text": "second"}
                ).status_code == 503
            finally:
                release.set()
            assert future.result(timeout=5).status_code == 200
        assert client.post(
            "/api/embeddings/visual", headers=headers, json={"text": "third"}
        ).status_code == 200


def test_embedding_failed_startup_never_reports_ready(embedding_module, monkeypatch):
    module, _model = embedding_module

    def fail(_path):
        raise RuntimeError("weights unavailable")

    monkeypatch.setattr(module, "load_model", fail)
    app = module.create_app()
    with pytest.raises(RuntimeError, match="weights unavailable"), TestClient(app):
        pass
    assert app.state.ready is False


def test_embedding_rejects_invalid_vectors(embedding_module):
    module, _model = embedding_module

    class InvalidVector:
        def tolist(self):
            return [float("nan")] * 2048

    class InvalidModel:
        def process(self, _items):
            return [InvalidVector()]

    with pytest.raises(RuntimeError, match="invalid embedding"):
        module.encode(InvalidModel(), {"text": "check"})


def test_embedding_compose_keeps_gpu_private_and_reid_space_unchanged():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    services = yaml.safe_load(
        (root / "deploy/containers/compose.embedding.yaml").read_text()
    )["services"]
    embedding = services["embedding"]
    assert embedding["profiles"] == ["embedding"]
    assert "ports" not in embedding
    assert embedding["read_only"] is True
    assert embedding["volumes"][0].endswith(":/model:ro")
    assert embedding["environment"]["DATABASE_URL"] == "sqlite:///:memory:"
    assert embedding["environment"]["REID_ENABLED"] == "false"
    settings = services["api"]["environment"]
    assert settings["VISUAL_EMBEDDING_DIM"] == "2048"
    assert settings["VISUAL_EMBEDDING_PROVIDER"] == "qwen3_vl_http"
    assert "MILVUS_VISUAL_COLLECTION_PREFIX" in settings
    assert "VISUAL_EMBEDDING_UPSTREAM_API_KEY" in settings
    assert "VISUAL_EMBEDDING_SERVICE_API_KEY" not in settings
    assert "MILVUS_COLLECTION_PREFIX" not in settings
    assert not any(key.startswith("REID_") for key in settings)


@pytest.mark.parametrize("input_type", ["text", "image"])
def test_upstream_embedding_key_is_independent_of_public_auth(
    monkeypatch, tmp_path, input_type
):
    load_app(
        monkeypatch, tmp_path, "upstream-auth",
        VISUAL_EMBEDDING_PROVIDER="qwen3_vl_http",
        VISUAL_EMBEDDING_SERVICE_URL="http://embedding.invalid:18032",
        VISUAL_EMBEDDING_SERVICE_API_KEY="public-key",
        VISUAL_EMBEDDING_UPSTREAM_API_KEY="internal-key",
        VISUAL_EMBEDDING_DIM="2",
    )
    from app.config.settings import get_settings
    from app.services import embeddings

    captured = []

    def request(_request, timeout):
        captured.append(_request.get_header("Authorization"))
        return io.BytesIO(b'{"embedding": [1.0, 0.0]}')

    monkeypatch.setattr(embeddings.request, "urlopen", request)
    settings = get_settings()
    service = embeddings.VisualEmbeddingService(settings)
    if input_type == "image":
        image = tmp_path / "test.png"
        image.write_bytes(b"test payload")
        assert service.embed_image(image) == [1.0, 0.0]
    else:
        assert service.embed_text("test") == [1.0, 0.0]
    assert captured == ["Bearer internal-key"]
    assert settings.visual_embedding_service_api_key == "public-key"
