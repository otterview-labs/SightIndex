"""Acceptance checks use mocked GETs and actual handlers on generated media only."""

from __future__ import annotations

import copy
import json
import sqlite3
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from urllib import error

import pytest
from sqlalchemy import event

from app.services.index_identity import (
    QWEN_EMBEDDING_DIM,
    QWEN_MAX_LENGTH,
    QWEN_MAX_PIXELS,
    QWEN_MODEL_ID,
    QWEN_RUNTIME_SHA256,
    QWEN_WEIGHTS_SHA256,
    milvus_namespace_identity,
    reid_index_fingerprint,
)
from deploy.containers import verify


@pytest.fixture
def settings(tmp_path: Path) -> Any:
    """Explicit settings prevent a developer's deployment .env from driving tests."""
    from app.config.settings import Settings

    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'fixture.sqlite'}",
        data_dir=tmp_path / "fixture-data",
        app_basic_auth_username="fixture-user",
        app_basic_auth_password="fixture-password",
        reid_face_priority_enabled=True,
        reid_model="fixture-reid",
        reid_checkpoint_revision="fixture-revision",
        reid_preprocess_version="fixture-preprocessing",
        reid_embedding_dim=32,
        visual_embedding_provider="qwen3_vl_http",
        visual_embedding_service_url="http://embedding.test:18032",
        visual_embedding_upstream_api_key="private-fixture-upstream-key",
        visual_embedding_service_api_key="private-fixture-api-key",
        visual_embedding_model=QWEN_MODEL_ID,
        visual_embedding_dim=QWEN_EMBEDDING_DIM,
        vlm_provider="none",
    )


@pytest.fixture
def openapi() -> dict[str, Any]:
    """Use the actual API schemas without creating the production application."""
    from fastapi import FastAPI

    from app.api.media import router as media_router
    from app.api.reid import router as reid_router

    api = FastAPI()
    api.include_router(media_router, prefix="/api")
    api.include_router(reid_router, prefix="/api")
    return api.openapi()


@pytest.fixture
def reid(settings: Any) -> dict[str, Any]:
    """Return a ready status with a complete model identity and pending coverage."""
    return {
        "enabled": True,
        "ready": True,
        "reid_service_ok": True,
        "milvus_configured": True,
        "milvus_ok": True,
        "milvus_in_cooldown": False,
        "last_error": None,
        "model": settings.reid_model,
        "checkpoint_revision": settings.reid_checkpoint_revision,
        "embedding_dim": settings.reid_embedding_dim,
        "preprocess_version": settings.reid_preprocess_version,
        "milvus_namespace": milvus_namespace_identity(settings),
        "index_fingerprint": reid_index_fingerprint(settings),
        "face_priority_enabled": True,
        "face_priority_ready": True,
        "indexed_crops": 3,
        "pending_crops": 7,
    }


def test_actual_openapi_contract(openapi: dict[str, Any]) -> None:
    verify.validate_openapi_contract(openapi)


@pytest.mark.parametrize("path", verify.PLAYBACK_PATHS)
def test_missing_playback_route_fails(openapi: dict[str, Any], path: str) -> None:
    del openapi["paths"][path]
    with pytest.raises(verify.AcceptanceError, match="must be an object"):
        verify.validate_openapi_contract(openapi)


@pytest.mark.parametrize("schema", ["VideoPlaybackRead", "ReidFaceCoverage", "ReidSearchResponse"])
def test_missing_schema_fails(openapi: dict[str, Any], schema: str) -> None:
    del openapi["components"]["schemas"][schema]
    with pytest.raises(verify.AcceptanceError, match="must be an object"):
        verify.validate_openapi_contract(openapi)


@pytest.mark.parametrize(
    ("schema", "field"),
    [
        ("VideoPlaybackRead", "available"),
        ("VideoPlaybackRead", "source_type"),
        ("VideoPlaybackRead", "video_url"),
        ("VideoPlaybackRead", "offset_seconds"),
        ("VideoPlaybackRead", "captured_at"),
        ("VideoPlaybackRead", "reason"),
        ("ReidFaceCoverage", "compared_count"),
        ("ReidFaceCoverage", "query_identity_verified"),
        ("ReidFaceCoverage", "query_absence_reasons"),
    ],
)
def test_schema_field_types_are_validated(openapi: dict[str, Any], schema: str, field: str) -> None:
    openapi["components"]["schemas"][schema]["properties"][field] = {"type": "array"}
    with pytest.raises(verify.AcceptanceError, match="invalid schema"):
        verify.validate_openapi_contract(openapi)


def test_wrong_playback_response_and_coverage_reference_fail(openapi: dict[str, Any]) -> None:
    original = copy.deepcopy(openapi)
    operation = openapi["paths"][verify.PLAYBACK_PATHS[0]]["get"]
    operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"] = "wrong"
    with pytest.raises(verify.AcceptanceError, match="must return VideoPlaybackRead"):
        verify.validate_openapi_contract(openapi)
    original["components"]["schemas"]["ReidSearchResponse"]["properties"]["face_coverage"] = {}
    with pytest.raises(verify.AcceptanceError, match="must expose ReidFaceCoverage"):
        verify.validate_openapi_contract(original)


def test_required_playback_fields(openapi: dict[str, Any]) -> None:
    openapi["components"]["schemas"]["VideoPlaybackRead"]["required"] = []
    with pytest.raises(verify.AcceptanceError, match="requires available and source_type"):
        verify.validate_openapi_contract(openapi)


@pytest.mark.parametrize("value", [-1, True, False, 1.0, "1", None])
def test_invalid_counts_fail(value: Any) -> None:
    with pytest.raises(verify.AcceptanceError, match="nonnegative integer"):
        verify.validate_media_counts({"image_with_crops_count": value, "person_crop_count": 0})


def test_valid_counts_do_not_print_ids_or_pixels() -> None:
    assert "0 images" in verify.validate_media_counts(
        {"image_with_crops_count": 0, "person_crop_count": 1}
    )


def test_basic_auth_and_port_configuration(settings: Any) -> None:
    http = verify.application_http(settings, {"APP_PORT": "8123"})
    assert http.base_url == "http://127.0.0.1:8123"
    assert http.authorization == "Basic Zml4dHVyZS11c2VyOmZpeHR1cmUtcGFzc3dvcmQ="
    assert "fixture-password" not in repr(http)
    assert "Basic" not in repr(http)
    default = verify.application_http(settings, {})
    assert default.base_url == "http://127.0.0.1:8000"
    assert verify.application_http(
        settings, {"API_BASE_URL": "http://api.test:9000/"}
    ).base_url == ("http://api.test:9000")


@pytest.mark.parametrize("value", ["bad", "0", "65536"])
def test_invalid_api_port_fails(settings: Any, value: str) -> None:
    with pytest.raises(verify.AcceptanceError, match="APP_PORT"):
        verify.application_http(settings, {"APP_PORT": value})


@pytest.mark.parametrize(
    "url",
    [
        "file:///secret",
        "http://user:secret@api.test",
        "http://api.test?key=secret",
        "http://api.test#secret",
        "http://api.test:invalid",
    ],
)
def test_unsafe_base_url_is_not_echoed(settings: Any, url: str) -> None:
    with pytest.raises(verify.AcceptanceError) as failure:
        verify.application_http(settings, {"API_BASE_URL": url})
    assert "secret" not in str(failure.value)


def test_incomplete_basic_auth_fails(settings: Any) -> None:
    settings.app_basic_auth_password = None
    with pytest.raises(verify.AcceptanceError, match="both Basic auth"):
        verify.application_http(settings, {})
    settings.app_basic_auth_password = "fixture-password"
    settings.app_basic_auth_username = "bad:user"
    with pytest.raises(verify.AcceptanceError, match="colon"):
        verify.application_http(settings, {})
    settings.app_basic_auth_username = None
    settings.app_basic_auth_password = None
    assert verify.application_http(settings, {}).authorization is None


def test_readonly_http_uses_bounded_get_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    @contextmanager
    def response(req: Any, timeout: float) -> Any:
        seen.append((req, timeout))

        class Response:
            status = 200

            def read(self, limit: int) -> bytes:
                assert limit == 8 * 1024 * 1024 + 1
                return b'{"count": 1}'

        yield Response()

    class Opener:
        open = staticmethod(response)

    monkeypatch.setattr(verify.request, "build_opener", lambda *_handlers: Opener())
    assert verify.ReadOnlyHTTP("http://api.test", "Basic fixture").get_json("/counts") == {
        "count": 1
    }
    req, timeout = seen[0]
    assert req.method == "GET"
    assert req.headers["Authorization"] == "Basic fixture"
    assert req.data is None
    assert timeout == 15
    assert verify._NoRedirect().redirect_request(None, None, 302, "", None, "") is None


@pytest.mark.parametrize(
    "kind", ["auth", "unreachable", "bad-json", "array", "oversized", "status"]
)
def test_readonly_http_errors_are_sanitized(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    @contextmanager
    def response(_req: Any, timeout: float) -> Any:
        assert timeout > 0
        if kind == "auth":
            raise error.HTTPError("http://secret.test", 401, "secret-body", {}, None)
        if kind == "unreachable":
            raise error.URLError("secret-network-error")

        class Response:
            status = 503 if kind == "status" else 200

            def read(self, _limit: int) -> bytes:
                if kind == "oversized":
                    return b"x" * (8 * 1024 * 1024 + 1)
                return b"[1]" if kind == "array" else b"secret-invalid-json"

        yield Response()

    class Opener:
        open = staticmethod(response)

    monkeypatch.setattr(verify.request, "build_opener", lambda *_handlers: Opener())
    with pytest.raises(verify.AcceptanceError) as failure:
        verify.ReadOnlyHTTP("http://api.test", "private-key").get_json("/counts")
    assert "secret" not in str(failure.value)
    assert "private-key" not in str(failure.value)


def _database(path: Path, columns: str) -> None:
    """Create only a synthetic schema and one fixture row."""
    with sqlite3.connect(path) as db:
        db.execute(f"CREATE TABLE images (id TEXT PRIMARY KEY, {columns})")
        db.execute("INSERT INTO images (id) VALUES ('synthetic-row')")


def test_database_inspection_never_reads_rows_or_writes(
    monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    from sqlalchemy.engine import make_url

    path = Path(make_url(settings.database_url).database)
    _database(path, "processed_at DATETIME, source_video_url TEXT, video_offset_seconds FLOAT")
    before = path.read_bytes()
    statements: list[str] = []
    connect = sqlite3.connect

    def tracked_connect(database_uri: str, *, uri: bool) -> Any:
        assert database_uri.endswith("?mode=ro")
        connection = connect(database_uri, uri=uri)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(verify.sqlite3, "connect", tracked_connect)
    assert verify.inspect_database_contract(settings) == verify.IMAGE_COLUMNS
    assert path.read_bytes() == before
    assert statements
    assert not any("FROM images" in statement for statement in statements)
    assert not any(
        statement.upper().startswith(("INSERT", "UPDATE", "ALTER", "CREATE", "DELETE"))
        for statement in statements
    )


def test_database_missing_columns_and_missing_file_fail(settings: Any) -> None:
    from sqlalchemy.engine import make_url

    path = Path(make_url(settings.database_url).database)
    with pytest.raises(verify.AcceptanceError, match="does not exist"):
        verify.inspect_database_contract(settings)
    assert not path.exists()
    _database(path, "processed_at DATETIME")
    with pytest.raises(verify.AcceptanceError, match="source_video_url, video_offset_seconds"):
        verify.inspect_database_contract(settings)
    settings.database_url = "sqlite:///:memory:"
    with pytest.raises(verify.AcceptanceError, match="existing file"):
        verify.inspect_database_contract(settings)
    settings.database_url = "mysql://fixture@database.test/fixture"
    with pytest.raises(verify.AcceptanceError, match="SQLite and PostgreSQL"):
        verify.inspect_database_contract(settings)


def test_ready_reid_reports_pending_without_backfill(settings: Any, reid: dict[str, Any]) -> None:
    assert "7 pending" in verify.validate_reid_status(reid, settings)
    settings.reid_face_priority_enabled = False
    reid["face_priority_enabled"] = False
    reid["face_priority_ready"] = False
    assert "face priority disabled" in verify.validate_reid_status(reid, settings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ready", False),
        ("enabled", False),
        ("reid_service_ok", False),
        ("milvus_configured", False),
        ("milvus_ok", False),
        ("milvus_in_cooldown", True),
        ("last_error", "secret-upstream-url"),
        ("model", "other-model"),
        ("embedding_dim", True),
        ("checkpoint_revision", ""),
        ("index_fingerprint", ""),
        ("index_fingerprint", "nonempty-but-stale-space"),
        ("milvus_namespace", "wrong-nonempty-namespace"),
        ("face_priority_enabled", False),
        ("face_priority_ready", False),
        ("pending_crops", True),
    ],
)
def test_reid_failures_are_sanitized(
    settings: Any, reid: dict[str, Any], field: str, value: Any
) -> None:
    reid[field] = value
    with pytest.raises(verify.AcceptanceError) as failure:
        verify.validate_reid_status(reid, settings)
    assert "secret" not in str(failure.value)


def test_embedding_private_key_and_endpoint_normalization(settings: Any) -> None:
    settings.visual_embedding_service_url += "/api/embeddings/visual"
    http = verify.embedding_http(settings)
    assert http.base_url == "http://embedding.test:18032"
    assert http.authorization == "Bearer private-fixture-upstream-key"
    assert "private-fixture" not in repr(http)
    assert "ready" in verify.validate_embedding_health(_embedding_health(), settings)


def _embedding_health() -> dict[str, Any]:
    """Return the complete reviewed identity, not merely a healthy model name."""
    return {
        "ready": True,
        "model": QWEN_MODEL_ID,
        "dim": QWEN_EMBEDDING_DIM,
        "weights_sha256": QWEN_WEIGHTS_SHA256,
        "runtime_sha256": QWEN_RUNTIME_SHA256,
        "max_pixels": QWEN_MAX_PIXELS,
        "max_length": QWEN_MAX_LENGTH,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ready", False),
        ("model", "wrong"),
        ("dim", True),
        ("dim", 2049),
        ("weights_sha256", "private-mismatched-hash"),
        ("runtime_sha256", "private-mismatched-runtime"),
        ("max_pixels", QWEN_MAX_PIXELS + 1),
        ("max_pixels", float(QWEN_MAX_PIXELS)),
        ("max_length", QWEN_MAX_LENGTH + 1),
        ("max_length", True),
    ],
)
def test_embedding_health_requires_live_ready_and_matching_identity(
    settings: Any, field: str, value: Any
) -> None:
    payload = _embedding_health()
    payload[field] = value
    with pytest.raises(verify.AcceptanceError) as failure:
        verify.validate_embedding_health(payload, settings)
    assert "private-" not in str(failure.value)


@pytest.mark.parametrize("field", list(_embedding_health()))
def test_legacy_embedding_health_missing_identity_fails(settings: Any, field: str) -> None:
    payload = _embedding_health()
    del payload[field]
    with pytest.raises(verify.AcceptanceError):
        verify.validate_embedding_health(payload, settings)


@pytest.mark.parametrize(
    ("field", "value"), [("visual_embedding_model", "unreviewed"), ("visual_embedding_dim", 8)]
)
def test_embedding_configuration_cannot_relax_reviewed_identity(
    settings: Any, field: str, value: Any
) -> None:
    setattr(settings, field, value)
    payload = _embedding_health()
    payload["model" if field == "visual_embedding_model" else "dim"] = value
    with pytest.raises(verify.AcceptanceError):
        verify.validate_embedding_health(payload, settings)


def test_embedding_health_reports_complete_identity_from_actual_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings: Any
) -> None:
    """Only mocked model startup is used; no local/remote weights are accessed."""
    from fastapi.testclient import TestClient

    from deploy.containers import embedding_app

    local = embedding_app.EmbeddingSettings(
        _env_file=None,
        visual_embedding_service_api_key="synthetic-health-test-key",
        visual_embedding_model=str(tmp_path / "unused-synthetic-model"),
    )
    monkeypatch.setattr(embedding_app, "EmbeddingSettings", lambda: local)
    monkeypatch.setattr(embedding_app, "load_model", lambda _path: object())
    monkeypatch.setattr(embedding_app, "encode", lambda _model, _item: [1.0])
    with TestClient(embedding_app.create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == _embedding_health()
        assert "ready" in verify.validate_embedding_health(response.json(), settings)


def test_embedding_configuration_failures(settings: Any) -> None:
    settings.visual_embedding_provider = "none"
    with pytest.raises(verify.AcceptanceError, match="provider"):
        verify.embedding_http(settings)
    settings.visual_embedding_provider = "qwen3_vl_http"
    settings.visual_embedding_service_url = None
    with pytest.raises(verify.AcceptanceError, match="URL is required"):
        verify.embedding_http(settings)
    settings.visual_embedding_service_url = "http://embedding.test"
    settings.visual_embedding_upstream_api_key = None
    assert verify.embedding_http(settings).authorization == "Bearer private-fixture-api-key"
    settings.visual_embedding_service_api_key = None
    with pytest.raises(verify.AcceptanceError, match="key is required"):
        verify.embedding_http(settings)


def _semantic(settings: Any) -> dict[str, Any]:
    return {
        "enabled": True,
        "configured": True,
        "model": settings.visual_embedding_model,
        "total_crops": 8,
        "indexed_crops": 2,
        "labeled_crops": 3,
    }


def test_semantic_reports_partial_coverage(settings: Any) -> None:
    assert "2/8 indexed, 6 pending" in verify.validate_semantic_status(
        _semantic(settings), settings
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("enabled", False), ("configured", False), ("model", "wrong"), ("indexed_crops", 9)],
)
def test_semantic_missing_configuration_fails(settings: Any, field: str, value: Any) -> None:
    payload = _semantic(settings)
    payload[field] = value
    with pytest.raises(verify.AcceptanceError):
        verify.validate_semantic_status(payload, settings)


def test_vlm_none_and_external_configuration_are_not_quality_claims(settings: Any) -> None:
    assert "VLM disabled" in verify.validate_vlm_configuration(settings)
    settings.vlm_provider = "openai_compatible"
    assert "not been verified" in verify.validate_vlm_configuration(settings)
    settings.vlm_model = ""
    with pytest.raises(verify.AcceptanceError, match="model"):
        verify.validate_vlm_configuration(settings)
    settings.vlm_provider = "unsupported"
    with pytest.raises(verify.AcceptanceError, match="unsupported"):
        verify.validate_vlm_configuration(settings)


def test_runtime_only_requests_readonly_endpoints(
    monkeypatch: pytest.MonkeyPatch, settings: Any, openapi: dict[str, Any], reid: dict[str, Any]
) -> None:
    called: list[str] = []
    payloads = {
        "/api/media/counts": {"image_with_crops_count": 0, "person_crop_count": 0},
        "/openapi.json": openapi,
        "/api/reid/status": reid,
        "/api/search/semantic/status": _semantic(settings),
        "/health": _embedding_health(),
    }

    def get_json(_self: Any, path: str) -> Mapping[str, Any]:
        called.append(path)
        return payloads[path]

    monkeypatch.setattr(verify.ReadOnlyHTTP, "get_json", get_json)
    monkeypatch.setattr(verify, "inspect_database_contract", lambda _settings: verify.IMAGE_COLUMNS)
    results = verify.verify_runtime(
        settings,
        frozenset({"base", "reid", "embedding", "semantic"}),
        verify.ReadOnlyHTTP("http://api.test"),
    )
    assert len(results) == 7
    assert all(result.passed for result in results)
    assert called == [
        "/api/media/counts",
        "/openapi.json",
        "/api/reid/status",
        "/health",
        "/api/search/semantic/status",
    ]
    reid["ready"] = False
    results = verify.verify_runtime(
        settings, frozenset({"base", "reid"}), verify.ReadOnlyHTTP("http://api.test")
    )
    assert results[-1].passed is False
    assert all(result.passed for result in results[:-1])


def test_unknown_exception_contents_are_never_printed() -> None:
    def fail() -> str:
        raise RuntimeError("private-key-in-response")

    result = verify._run_check("fixture", fail)
    assert not result.passed
    assert "private-key" not in result.detail


def test_actual_isolated_upload_smoke_never_touches_production_engine_or_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config.settings import get_settings
    from app.db import session

    settings = get_settings()
    before = settings.model_dump()
    temporary_paths: list[Path] = []

    def temporary_directory(*args: Any, **kwargs: Any) -> Any:
        temporary = TemporaryDirectory(*args, **kwargs)
        temporary_paths.append(Path(temporary.name))
        return temporary

    def forbid_production(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("acceptance smoke accessed the production database")

    monkeypatch.setattr(verify, "TemporaryDirectory", temporary_directory)
    event.listen(session.engine, "connect", forbid_production)
    event.listen(session.engine, "before_cursor_execute", forbid_production)
    try:
        assert "HTTP 206" in verify.isolated_upload_smoke()
    finally:
        event.remove(session.engine, "connect", forbid_production)
        event.remove(session.engine, "before_cursor_execute", forbid_production)
    assert settings.model_dump() == before
    assert temporary_paths and all(not path.exists() for path in temporary_paths)


def test_encoder_failure_is_not_silently_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cv2

    class BrokenWriter:
        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    monkeypatch.setattr(cv2, "VideoWriter", lambda *_args: BrokenWriter())
    with pytest.raises(verify.AcceptanceError, match="MJPG encoder is unavailable"):
        verify._write_synthetic_video(tmp_path / "synthetic.avi")


@pytest.mark.parametrize("offset", [True, -1, None, float("nan"), float("inf")])
def test_bad_playback_mapping_fails(offset: Any) -> None:
    with pytest.raises(verify.AcceptanceError, match="parent video/offset"):
        verify._validate_smoke_playback(
            200,
            {
                "available": True,
                "source_type": "video_frame",
                "video_url": "/data/videos/a.avi",
                "offset_seconds": offset,
            },
            "/data/videos/a.avi",
            0,
        )


def test_model_smoke_missing_assets_refuses_download(settings: Any, tmp_path: Path) -> None:
    settings.person_detector = "yolo"
    settings.yolo_model = str(tmp_path / "missing.pt")
    with pytest.raises(verify.AcceptanceError, match="downloads refused"):
        verify.model_smoke(settings)
    settings.person_detector = "yolo_service"
    with pytest.raises(verify.AcceptanceError, match="local"):
        verify.model_smoke(settings)
    settings.person_detector = "whole_frame"
    settings.face_insightface_root = tmp_path / "missing-face-assets"
    with pytest.raises(verify.AcceptanceError, match="downloads refused"):
        verify.model_smoke(settings)


@pytest.mark.parametrize("failure", [None, "provider", "fallback", "embedding", "image"])
def test_model_smoke_generated_pixels_and_no_downloads(
    monkeypatch: pytest.MonkeyPatch, settings: Any, tmp_path: Path, failure: str | None
) -> None:
    import cv2
    import numpy as np

    from app.face_algorithms import insightface_cuda
    from app.services import frame_processing

    directory = tmp_path / "models" / "fixture"
    directory.mkdir(parents=True)
    for filename in ("det_10g.onnx", "w600k_r50.onnx"):
        (directory / filename).touch()
    seen: list[str] = []

    class Recognizer:
        device = "cpu"
        model_name = "fixture"
        det_size = 640
        root = tmp_path
        providers = ["CPUExecutionProvider"]

        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["allow_download"] is False

        def model_dir(self) -> Path:
            return directory

        def info(self) -> Any:
            return SimpleNamespace(
                device="cpu", available_providers=[] if failure == "provider" else self.providers
            )

        def extract(self, path: Path) -> list[Any]:
            assert path.name == "synthetic.png"
            assert cv2.imread(str(path)).shape == (128, 128, 3)
            seen.append("extract")
            return []

    def detect(path: Path) -> list[Any]:
        assert path.name == "synthetic.png"
        seen.append("detect")
        return []

    def get_feat(pixels: Any) -> Any:
        assert pixels.shape == (112, 112, 3)
        assert (pixels == 96).all()
        seen.append("recognition")
        return np.zeros((1, 3 if failure == "embedding" else settings.face_embedding_dim))

    fake_model = SimpleNamespace(
        session=SimpleNamespace(
            get_providers=lambda: (
                ["CUDAExecutionProvider"] if failure == "fallback" else ["CPUExecutionProvider"]
            )
        ),
        get_feat=get_feat,
    )
    monkeypatch.setattr(insightface_cuda, "InsightFaceCudaRecognizer", Recognizer)
    monkeypatch.setattr(
        insightface_cuda,
        "_insightface_app",
        lambda *_args: SimpleNamespace(models={"recognition": fake_model}),
    )
    monkeypatch.setattr(
        frame_processing, "create_person_detector", lambda _settings: SimpleNamespace(detect=detect)
    )
    monkeypatch.setitem(sys.modules, "insightface.app", SimpleNamespace(FaceAnalysis=object))
    settings.person_detector = "whole_frame"
    settings.face_embedding_provider = "insightface"
    settings.face_embedding_dim = 512
    if failure == "image":
        monkeypatch.setattr(cv2, "imwrite", lambda *_args: False)
    if failure:
        with pytest.raises(verify.AcceptanceError):
            verify.model_smoke(settings)
    else:
        assert "do not establish model quality" in verify.model_smoke(settings)
        assert seen == ["detect", "extract", "recognition"]


@pytest.mark.parametrize("value", ["", "base fake", "base,reid"])
def test_invalid_stack_selection_fails(value: str) -> None:
    with pytest.raises(verify.AcceptanceError, match="--stacks"):
        verify.parse_stacks(value)


def test_stack_selection_always_includes_base() -> None:
    assert verify.parse_stacks("reid semantic") == frozenset({"base", "reid", "semantic"})


def test_cli_exit_status_and_secret_protection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        verify, "verify_runtime", lambda *_args: [verify.CheckResult("mock runtime", True, "safe")]
    )
    monkeypatch.setattr(verify, "isolated_upload_smoke", lambda: "mocked upload")
    monkeypatch.setattr(verify, "model_smoke", lambda _settings: "mocked model")
    assert verify.main(["--stacks", "base", "--upload-smoke", "--model-smoke"]) == 0
    output = capsys.readouterr().out
    assert "[PASS] upload smoke" in output
    assert "[PASS] model smoke" in output
    monkeypatch.setattr(
        verify,
        "verify_runtime",
        lambda *_args: [verify.CheckResult("mock runtime", False, "failed")],
    )
    assert verify.main([]) == 1
    assert verify.main(["--stacks", "invalid"]) == 1
    assert "--stacks" in capsys.readouterr().err


def test_invalid_payload_object_fails() -> None:
    with pytest.raises(verify.AcceptanceError, match="object"):
        verify._mapping(json.loads("[]"), "fixture")
