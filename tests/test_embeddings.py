import json
import sys
from urllib import error, request

from app.config.settings import Settings
from app.services import embeddings
from app.services.embeddings import EmbeddingRuntimeError, VisualEmbeddingService
from app.services.rerank import VLMRerankService


class _FakeHTTPResponse:
    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps({"embedding": [0.25, 0.75], "dim": 2}).encode("utf-8")


class _FakeDashScopeHTTPResponse:
    def __enter__(self) -> "_FakeDashScopeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "output": {
                    "embeddings": [
                        {
                            "embedding": [0.5, 0.5],
                        }
                    ]
                }
            }
        ).encode("utf-8")


def test_qwen3_vl_http_provider_posts_to_visual_embedding_endpoint(monkeypatch):
    captured: dict[str, object] = {}

    def fake_urlopen(req: request.Request, timeout: int) -> _FakeHTTPResponse:
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["authorization"] = req.headers.get("Authorization")
        return _FakeHTTPResponse()

    monkeypatch.setattr("app.services.embeddings.request.urlopen", fake_urlopen)
    service = VisualEmbeddingService(
        Settings(
            visual_embedding_provider="qwen3_vl_http",
            visual_embedding_dim=2,
            visual_embedding_service_url="http://127.0.0.1:18021",
            visual_embedding_service_api_key="secret-token",
            visual_embedding_service_timeout_seconds=12,
        )
    )

    vector = service.embed_text("白色衣服")

    assert vector == [0.25, 0.75]
    assert captured["url"] == "http://127.0.0.1:18021/api/embeddings/visual"
    assert captured["timeout"] == 12
    assert captured["body"] == {
        "text": "白色衣服",
        "instruction": "Retrieve images that match the user query.",
    }
    assert captured["authorization"] == "Bearer secret-token"


def test_qwen3_vl_http_provider_fail_fast_during_cooldown(monkeypatch):
    calls = 0

    def fake_urlopen(req: request.Request, timeout: int) -> _FakeHTTPResponse:
        nonlocal calls
        calls += 1
        raise error.URLError("timed out")

    monkeypatch.setattr("app.services.embeddings.request.urlopen", fake_urlopen)
    monkeypatch.setattr(embeddings, "_HTTP_EMBEDDING_FAILURE_UNTIL", {})
    service = VisualEmbeddingService(
        Settings(
            visual_embedding_provider="qwen3_vl_http",
            visual_embedding_dim=2,
            visual_embedding_service_url="http://127.0.0.1:18021",
            visual_embedding_service_timeout_seconds=1,
            visual_embedding_service_failure_cooldown_seconds=30,
        )
    )

    for expected_message in ["request failed", "failure cooldown"]:
        try:
            service.embed_text("白色衣服")
        except EmbeddingRuntimeError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError("expected embedding request to fail")

    assert calls == 1


def test_vlm_rerank_service_posts_to_embedding_service_rerank_endpoint(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    image_path = tmp_path / "crop.jpg"
    image_path.write_bytes(b"image")

    class FakeRerankHTTPResponse:
        def __enter__(self) -> "FakeRerankHTTPResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"score": 0.82, "matched": True, "reason": "reranker score"}
            ).encode("utf-8")

    def fake_urlopen(req: request.Request, timeout: int) -> FakeRerankHTTPResponse:
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["authorization"] = req.headers.get("Authorization")
        return FakeRerankHTTPResponse()

    monkeypatch.setattr("app.services.rerank.request.urlopen", fake_urlopen)
    service = VLMRerankService(
        Settings(
            vlm_rerank_service_url="http://127.0.0.1:18021",
            vlm_rerank_service_api_key="secret-token",
            vlm_rerank_timeout_seconds=12,
        )
    )

    decision = service.rerank_image("光头的人", image_path, {"hair": "bald"})

    assert decision.score == 0.82
    assert decision.matched is True
    assert captured["url"] == "http://127.0.0.1:18021/api/embeddings/rerank"
    assert captured["timeout"] == 12
    assert captured["body"] == {
        "query": "光头的人",
        "image_base64": "aW1hZ2U=",
        "image_filename": "crop.jpg",
        "attributes": {"hair": "bald"},
    }
    assert captured["authorization"] == "Bearer secret-token"


def test_dashscope_multimodal_provider_posts_native_payload(monkeypatch):
    captured: dict[str, object] = {}

    def fake_urlopen(req: request.Request, timeout: int) -> _FakeDashScopeHTTPResponse:
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["authorization"] = req.headers.get("Authorization")
        return _FakeDashScopeHTTPResponse()

    monkeypatch.setattr("app.services.embeddings.request.urlopen", fake_urlopen)
    service = VisualEmbeddingService(
        Settings(
            visual_embedding_provider="dashscope_multimodal",
            visual_embedding_model="qwen3-vl-embedding",
            visual_embedding_dim=2,
            visual_embedding_service_api_key="dashscope-token",
            visual_embedding_service_timeout_seconds=15,
        )
    )

    vector = service.embed_text("白色衣服")

    assert vector == [0.5, 0.5]
    assert captured["url"] == (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "multimodal-embedding/multimodal-embedding"
    )
    assert captured["timeout"] == 15
    assert captured["authorization"] == "Bearer dashscope-token"
    assert captured["body"] == {
        "model": "qwen3-vl-embedding",
        "input": {
            "contents": [
                {
                    "text": "白色衣服",
                }
            ]
        },
        "parameters": {
            "dimension": 2,
        },
    }


def test_torchvision_compat_is_idempotent_when_namespace_exists(monkeypatch):
    class FakeLibrary:
        calls = 0

        def __init__(self, namespace: str, kind: str) -> None:
            self.namespace = namespace
            self.kind = kind
            FakeLibrary.calls += 1
            if FakeLibrary.calls > 1:
                raise RuntimeError("Only a single TORCH_LIBRARY can be used")

        def define(self, signature: str) -> None:
            self.signature = signature

    class FakeTorch:
        class library:
            Library = FakeLibrary

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(embeddings, "_TORCHVISION_COMPAT_READY", False)
    monkeypatch.setattr(embeddings, "_TORCHVISION_STUB_LIBS", [])

    embeddings._ensure_torchvision_import_compat()
    embeddings._ensure_torchvision_import_compat()

    assert FakeLibrary.calls == 1
    assert embeddings._TORCHVISION_COMPAT_READY is True
