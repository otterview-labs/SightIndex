"""Verify deployed contracts without changing production data or fetching models.

Run from the API container or the application virtual environment. The optional
upload smoke uses its own SQLite database and generated pixels; it never uploads
to the deployed HTTP service. Model smoke requires already provisioned assets.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from urllib import error, parse, request

if TYPE_CHECKING:
    from app.config.settings import Settings

PLAYBACK_PATHS = ("/api/images/{image_id}/playback", "/api/person-crops/{crop_id}/playback")
IMAGE_COLUMNS = ("processed_at", "source_video_url", "video_offset_seconds")
STACKS = frozenset({"base", "reid", "embedding", "semantic"})


class AcceptanceError(RuntimeError):
    """A safe diagnostic containing no credential or arbitrary server response."""


@dataclass(frozen=True)
class CheckResult:
    """One independent acceptance result safe to print to the terminal."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ReadOnlyHTTP:
    """Bounded JSON GET client with explicit auth and no redirects or proxies."""

    base_url: str
    authorization: str | None = dataclass_field(default=None, repr=False)
    timeout: float = 15.0

    def get_json(self, path: str) -> Mapping[str, Any]:
        """Read a JSON object without printing secrets or remote response bodies."""
        headers = {"Accept": "application/json"}
        if self.authorization:
            headers["Authorization"] = self.authorization
        req = request.Request(self.base_url.rstrip("/") + path, headers=headers, method="GET")
        opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(req, timeout=self.timeout) as response:
                if response.status != 200:
                    raise AcceptanceError(f"GET {path} returned HTTP {response.status}")
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise AcceptanceError(f"GET {path} returned an oversized response")
            payload = json.loads(raw)
        except error.HTTPError as exc:
            raise AcceptanceError(f"GET {path} returned HTTP {exc.code}") from None
        except (OSError, error.URLError):
            raise AcceptanceError(f"GET {path} could not connect") from None
        except (ValueError, UnicodeError):
            raise AcceptanceError(f"GET {path} returned invalid JSON") from None
        return _mapping(payload, path)


class _NoRedirect(request.HTTPRedirectHandler):
    """Do not forward a Basic/Bearer credential to a redirected endpoint."""

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    """Require an object without reproducing untrusted values in the error."""
    if not isinstance(value, Mapping):
        raise AcceptanceError(f"{field} must be an object")
    return value


def _count(payload: Mapping[str, Any], field: str) -> int:
    """Validate a JSON count, rejecting bools as well as negative numbers."""
    value = payload.get(field)
    if type(value) is not int or value < 0:
        raise AcceptanceError(f"{field} must be a nonnegative integer")
    return value


def _nonempty(payload: Mapping[str, Any], field: str) -> str:
    """Require a nonempty identity string."""
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AcceptanceError(f"{field} must be a nonempty string")
    return value


def _url(value: str, field: str) -> str:
    """Validate service URLs without accepting credentials or query secrets."""
    parsed = parse.urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise AcceptanceError(f"{field} must be an HTTP URL without credentials/query/fragment")
    try:
        _port = parsed.port
    except ValueError:
        raise AcceptanceError(f"{field} has an invalid port") from None
    return value.rstrip("/")


def application_http(settings: Settings, env: Mapping[str, str]) -> ReadOnlyHTTP:
    """Use local APP_PORT or an explicit API base, with configured Basic auth."""
    base = env.get("SIGHTINDEX_API_BASE_URL") or env.get("API_BASE_URL")
    if not base:
        try:
            port = int(env.get("APP_PORT", "8000"))
        except ValueError:
            raise AcceptanceError("APP_PORT must be an integer") from None
        if not 1 <= port <= 65535:
            raise AcceptanceError("APP_PORT must be between 1 and 65535")
        base = f"http://127.0.0.1:{port}"
    username = settings.app_basic_auth_username
    password = settings.app_basic_auth_password
    if bool(username) != bool(password):
        raise AcceptanceError("both Basic auth username and password must be configured")
    authorization = None
    if username and password:
        if ":" in username:
            raise AcceptanceError("Basic auth username cannot contain a colon")
        token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        authorization = f"Basic {token}"
    return ReadOnlyHTTP(_url(base, "API base"), authorization)


def validate_media_counts(payload: Mapping[str, Any]) -> str:
    """Check database-backed aggregate counts without requesting media rows."""
    images = _count(payload, "image_with_crops_count")
    crops = _count(payload, "person_crop_count")
    return f"database-backed counts: {images} images with crops, {crops} crops"


def _schema_types(schema: Mapping[str, Any]) -> set[str]:
    """Read both nullable OpenAPI schema representations."""
    types = {schema["type"]} if isinstance(schema.get("type"), str) else set()
    for branch in schema.get("anyOf", []):
        types.update(_schema_types(_mapping(branch, "schema anyOf")))
    return types


def validate_openapi_contract(payload: Mapping[str, Any]) -> None:
    """Require stored-video playback routes and per-request face coverage fields.

    Raises:
        AcceptanceError: A route, response reference, schema, or field is missing.
    """
    paths = _mapping(payload.get("paths"), "OpenAPI paths")
    components = _mapping(payload.get("components"), "OpenAPI components")
    schemas = _mapping(components.get("schemas"), "OpenAPI schemas")
    for path in PLAYBACK_PATHS:
        operation = _mapping(_mapping(paths.get(path), path).get("get"), f"{path} GET")
        responses = _mapping(operation.get("responses"), f"{path} responses")
        success = _mapping(responses.get("200"), f"{path} response 200")
        content = _mapping(success.get("content"), f"{path} content")
        media = _mapping(content.get("application/json"), f"{path} JSON response")
        schema = _mapping(media.get("schema"), f"{path} response schema")
        if schema.get("$ref") != "#/components/schemas/VideoPlaybackRead":
            raise AcceptanceError(f"{path} must return VideoPlaybackRead")
    playback = _mapping(schemas.get("VideoPlaybackRead"), "VideoPlaybackRead")
    properties = _mapping(playback.get("properties"), "VideoPlaybackRead properties")
    for field, expected in (
        ("available", "boolean"),
        ("source_type", "string"),
        ("video_url", "string"),
        ("offset_seconds", "number"),
        ("captured_at", "string"),
        ("reason", "string"),
    ):
        if expected not in _schema_types(_mapping(properties.get(field), field)):
            raise AcceptanceError(f"VideoPlaybackRead.{field} has an invalid schema")
    if not {"available", "source_type"} <= set(playback.get("required", [])):
        raise AcceptanceError("VideoPlaybackRead requires available and source_type")
    coverage = _mapping(schemas.get("ReidFaceCoverage"), "ReidFaceCoverage")
    coverage_props = _mapping(coverage.get("properties"), "ReidFaceCoverage properties")
    for field, expected in (
        ("status", "string"),
        ("query_face_found", "boolean"),
        ("query_identity_verified", "boolean"),
        ("query_attempted_count", "integer"),
        ("candidate_attempted_count", "integer"),
        ("compared_count", "integer"),
        ("query_absence_reasons", "object"),
        ("candidate_absence_reasons", "object"),
    ):
        if expected not in _schema_types(_mapping(coverage_props.get(field), field)):
            raise AcceptanceError(f"ReidFaceCoverage.{field} has an invalid schema")
    search = _mapping(schemas.get("ReidSearchResponse"), "ReidSearchResponse")
    search_props = _mapping(search.get("properties"), "ReidSearchResponse properties")
    if _mapping(search_props.get("face_coverage"), "face_coverage").get("$ref") != (
        "#/components/schemas/ReidFaceCoverage"
    ):
        raise AcceptanceError("ReidSearchResponse must expose ReidFaceCoverage")


def inspect_database_contract(settings: Settings | None = None) -> tuple[str, ...]:
    """Inspect only images metadata using an explicitly read-only connection.

    SQLite uses mode=ro so a typo cannot create a database. PostgreSQL introspection
    runs in a read-only transaction. This does not initialize or migrate schemas.
    """
    from sqlalchemy import create_engine, inspect, text
    from sqlalchemy.engine import make_url

    if settings is None:
        from app.config.settings import Settings

        settings = Settings()
    url = make_url(settings.database_url)
    if url.get_backend_name() == "sqlite":
        if not url.database or url.database == ":memory:":
            raise AcceptanceError("deployed SQLite database must be an existing file")
        database = Path(url.database).resolve()
        if not database.is_file():
            raise AcceptanceError("deployed SQLite database file does not exist")
        engine = create_engine(
            "sqlite://",
            creator=lambda: sqlite3.connect(database.as_uri() + "?mode=ro", uri=True),
        )
    elif url.get_backend_name() == "postgresql":
        engine = create_engine(url, connect_args={"connect_timeout": 10})
    else:
        raise AcceptanceError("read-only inspection supports SQLite and PostgreSQL")
    try:
        with engine.connect() as connection, connection.begin():
            if engine.dialect.name == "postgresql":
                connection.execute(text("SET TRANSACTION READ ONLY"))
                connection.execute(text("SET LOCAL statement_timeout = '10s'"))
            columns = {column["name"] for column in inspect(connection).get_columns("images")}
        if not set(IMAGE_COLUMNS) <= columns:
            missing = ", ".join(sorted(set(IMAGE_COLUMNS) - columns))
            raise AcceptanceError(f"images is missing deployment columns: {missing}")
        return IMAGE_COLUMNS
    finally:
        engine.dispose()


def validate_reid_status(payload: Mapping[str, Any], settings: Settings) -> str:
    """Require live model/Milvus readiness; face priority is checked when enabled."""
    from app.services.index_identity import milvus_namespace_identity, reid_index_fingerprint

    for field in ("enabled", "ready", "reid_service_ok", "milvus_configured", "milvus_ok"):
        if payload.get(field) is not True:
            raise AcceptanceError(f"ReID {field} must be true")
    if payload.get("milvus_in_cooldown") is not False or payload.get("last_error"):
        raise AcceptanceError("ReID reports a cooldown or runtime error")
    for field, expected in (
        ("model", settings.reid_model),
        ("checkpoint_revision", settings.reid_checkpoint_revision),
        ("embedding_dim", settings.reid_embedding_dim),
        ("preprocess_version", settings.reid_preprocess_version),
    ):
        if not expected or payload.get(field) != expected or isinstance(payload.get(field), bool):
            raise AcceptanceError(f"ReID {field} does not match configured identity")
    for field, expected in (
        ("milvus_namespace", milvus_namespace_identity(settings)),
        ("index_fingerprint", reid_index_fingerprint(settings)),
    ):
        if _nonempty(payload, field) != expected:
            raise AcceptanceError(f"ReID {field} does not match configured identity")
    if payload.get("face_priority_enabled") is not settings.reid_face_priority_enabled:
        raise AcceptanceError("ReID face priority flag does not match configuration")
    if settings.reid_face_priority_enabled and payload.get("face_priority_ready") is not True:
        raise AcceptanceError("enabled ReID face priority is not ready")
    indexed = _count(payload, "indexed_crops")
    pending = _count(payload, "pending_crops")
    face = "ready" if settings.reid_face_priority_enabled else "disabled"
    return (
        f"ReID runtime/model/Milvus ready; face priority {face}; "
        f"{indexed} indexed, {pending} pending"
    )


def embedding_http(settings: Settings) -> ReadOnlyHTTP:
    """Use the configured embedding service and its private upstream key."""
    if settings.visual_embedding_provider.lower() not in {"qwen3_vl_http", "qwen3-vl-http"}:
        raise AcceptanceError("embedding acceptance requires the qwen3_vl_http provider")
    if not settings.visual_embedding_service_url:
        raise AcceptanceError("VISUAL_EMBEDDING_SERVICE_URL is required")
    base = _url(settings.visual_embedding_service_url, "embedding service URL")
    parsed = parse.urlsplit(base)
    # The embedding provider accepts either a service root or the concrete API endpoint.
    endpoint = "/api/embeddings/visual"
    if parsed.path.endswith(endpoint):
        base = base[: -len(endpoint)]
    key = settings.visual_embedding_upstream_api_key or settings.visual_embedding_service_api_key
    if not key:
        raise AcceptanceError("embedding upstream API key is required")
    return ReadOnlyHTTP(base, f"Bearer {key}", 30.0)


def validate_embedding_health(payload: Mapping[str, Any], settings: Settings) -> str:
    """Verify the embedding service's live startup inference and identity."""
    from app.services.index_identity import (
        QWEN_EMBEDDING_DIM,
        QWEN_MAX_LENGTH,
        QWEN_MAX_PIXELS,
        QWEN_MODEL_ID,
        QWEN_RUNTIME_SHA256,
        QWEN_WEIGHTS_SHA256,
    )

    if payload.get("ready") is not True:
        raise AcceptanceError("embedding service is not ready")
    if settings.visual_embedding_model != QWEN_MODEL_ID or payload.get("model") != QWEN_MODEL_ID:
        raise AcceptanceError("embedding model does not match configuration")
    if (
        settings.visual_embedding_dim != QWEN_EMBEDDING_DIM
        or _count(payload, "dim") != QWEN_EMBEDDING_DIM
    ):
        raise AcceptanceError("embedding dimension does not match configuration")
    for field, expected in (
        ("weights_sha256", QWEN_WEIGHTS_SHA256),
        ("runtime_sha256", QWEN_RUNTIME_SHA256),
        ("max_pixels", QWEN_MAX_PIXELS),
        ("max_length", QWEN_MAX_LENGTH),
    ):
        value = (
            _count(payload, field) if field in {"max_pixels", "max_length"} else payload.get(field)
        )
        if value != expected:
            raise AcceptanceError(f"embedding {field} does not match reviewed identity")
    return (
        "embedding ready with reviewed model/weights/runtime/preprocessing identity; "
        "quality has not been evaluated"
    )


def validate_semantic_status(payload: Mapping[str, Any], settings: Settings) -> str:
    """Check semantic configuration and describe coverage without backfilling."""
    if payload.get("enabled") is not True or payload.get("configured") is not True:
        raise AcceptanceError("requested semantic search must be enabled and configured")
    if payload.get("model") != settings.visual_embedding_model:
        raise AcceptanceError("semantic embedding model does not match configuration")
    total = _count(payload, "total_crops")
    indexed = _count(payload, "indexed_crops")
    labeled = _count(payload, "labeled_crops")
    if indexed > total or labeled > total:
        raise AcceptanceError("semantic coverage cannot exceed total crops")
    return (
        f"semantic enabled/configured; {indexed}/{total} indexed, {total - indexed} pending; "
        f"{labeled} labeled; historical backfill was not run"
    )


def validate_vlm_configuration(settings: Settings) -> str:
    """Report optional VLM honestly without generating text or incurring API cost."""
    provider = settings.vlm_provider.strip().lower()
    if provider in {"none", ""}:
        return "VLM disabled; automatic VLM labels/captions are not enabled"
    if provider != "openai_compatible":
        raise AcceptanceError("unsupported configured VLM provider")
    _url(settings.vlm_base_url, "VLM base URL")
    if not settings.vlm_model.strip():
        raise AcceptanceError("VLM model must be configured")
    return "VLM configuration present; endpoint inference and label quality have not been verified"


def _run_check(name: str, operation: Callable[[], str]) -> CheckResult:
    """Keep independent checks running and suppress arbitrary exception contents."""
    try:
        return CheckResult(name, True, operation())
    except AcceptanceError as exc:
        return CheckResult(name, False, str(exc))
    except Exception as exc:
        return CheckResult(
            name, False, f"check failed ({type(exc).__name__}); inspect local runtime"
        )


def verify_runtime(
    settings: Settings, stacks: frozenset[str], http: ReadOnlyHTTP
) -> list[CheckResult]:
    """Read deployed aggregate/status endpoints, schema, and database metadata."""

    def openapi() -> str:
        validate_openapi_contract(http.get_json("/openapi.json"))
        return "playback routes/schema and face coverage contract present"

    results = [
        _run_check(
            "media counts", lambda: validate_media_counts(http.get_json("/api/media/counts"))
        ),
        _run_check("OpenAPI", openapi),
        _run_check("database schema", lambda: ", ".join(inspect_database_contract(settings))),
        _run_check("VLM", lambda: validate_vlm_configuration(settings)),
    ]
    if "reid" in stacks:
        results.append(
            _run_check(
                "ReID", lambda: validate_reid_status(http.get_json("/api/reid/status"), settings)
            )
        )
    if "embedding" in stacks or "semantic" in stacks:
        results.append(
            _run_check(
                "embedding",
                lambda: validate_embedding_health(
                    embedding_http(settings).get_json("/health"), settings
                ),
            )
        )
    if "semantic" in stacks:
        results.append(
            _run_check(
                "semantic search",
                lambda: validate_semantic_status(
                    http.get_json("/api/search/semantic/status"), settings
                ),
            )
        )
    return results


def _write_synthetic_video(path: Path) -> None:
    """Encode five generated frames; absence of the encoder is an acceptance failure."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        writer.release()
        raise AcceptanceError("OpenCV MJPG encoder is unavailable")
    try:
        for index in range(5):
            pixels = np.full((48, 64, 3), 40 + index * 30, dtype=np.uint8)
            pixels[8:40, 16:48] = (180, 80 + index * 20, 30)
            writer.write(pixels)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise AcceptanceError("synthetic video encoder produced no bytes")


def verify_video_preparation_tools(settings: Settings) -> str:
    """Check opted-in local tools without opening media or reflecting tool output."""
    if not settings.video_preparation_enabled:
        return "video preparation disabled; no tools or stored media were inspected"
    for label, configured in (
        ("ffmpeg", settings.video_ffmpeg_binary),
        ("ffprobe", settings.video_ffprobe_binary),
    ):
        if not configured or any(character in configured for character in "\x00\r\n"):
            raise AcceptanceError(f"video preparation requires executable {label}")
        binary = shutil.which(configured)
        if binary is None:
            raise AcceptanceError(f"video preparation requires executable {label}")
        try:
            result = subprocess.run(
                [binary, "-version"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise AcceptanceError(f"video preparation tool check failed: {label}") from None
        if result.returncode != 0:
            raise AcceptanceError(f"video preparation tool check failed: {label}")
    return "ffmpeg and ffprobe executable; no media decoded or preparation accuracy claimed"


def isolated_upload_smoke() -> str:
    """Exercise actual upload/crop/playback/range handlers against temporary SQLite.

    The fresh FastAPI app has no production lifespan. Dependencies receive only
    the temporary settings/session; all model, indexing, recognition, and attribute
    work is disabled. Only TemporaryDirectory owns cleanup.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session, sessionmaker

    from app.api.media import router
    from app.api.media_files import MediaStaticFiles
    from app.config.settings import Settings, get_settings
    from app.db.session import Base, get_db
    from app.models.media import Image, PersonCrop
    from app.services.storage import StorageService

    with TemporaryDirectory(prefix="sightindex-acceptance-") as directory:
        temporary = Path(directory)
        settings = Settings(
            _env_file=None,
            database_url=f"sqlite:///{temporary / 'smoke.sqlite'}",
            data_dir=temporary / "data",
            video_preparation_enabled=False,
            person_detector="whole_frame",
            person_crop_require_whole_body=False,
            person_crop_min_confidence=0.0,
            person_crop_min_bbox_width=0,
            person_crop_min_bbox_height=0,
            person_crop_upscale_min_width=0,
            person_crop_upscale_min_height=0,
            person_crop_dedupe_enabled=False,
            appearance_tone_on_ingest=False,
            face_recognition_on_ingest=False,
            face_insightface_allow_download=False,
            vector_index_on_ingest=False,
            reid_enabled=False,
            reid_index_on_ingest=False,
            reid_face_priority_enabled=False,
            milvus_enabled=False,
            semantic_search_enabled=False,
            vlm_provider="none",
            vlm_structured_on_ingest=False,
            vlm_structured_background=False,
            vlm_caption_on_index=False,
            vlm_rerank_enabled=False,
            embedding_provider="none",
            visual_embedding_provider="none",
            stream_autostart_running=False,
        )
        StorageService(settings).ensure_dirs()
        video = temporary / "synthetic.avi"
        _write_synthetic_video(video)
        for name in ("persons", "events", "media", "vectors", "chat", "reid"):
            importlib.import_module(f"app.models.{name}")
        engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
        sessions = sessionmaker(bind=engine, expire_on_commit=False)

        def temporary_db() -> Generator[Session, None, None]:
            with sessions() as db:
                yield db

        api = FastAPI()
        api.dependency_overrides[get_db] = temporary_db
        api.dependency_overrides[get_settings] = lambda: settings
        api.include_router(router, prefix="/api")
        api.mount("/data", MediaStaticFiles(directory=settings.data_dir), name="data")
        try:
            Base.metadata.create_all(engine)
            with TestClient(api) as client, video.open("rb") as stream:
                response = client.post(
                    "/api/videos/upload?frame_interval_seconds=0.1&max_frames=5",
                    files={"file": ("synthetic.avi", stream, "video/x-msvideo")},
                )
                if response.status_code != 200:
                    raise AcceptanceError(f"isolated upload returned HTTP {response.status_code}")
                payload = _mapping(response.json(), "isolated upload")
                if _count(payload, "images_created") != 5 or _count(payload, "crops_created") < 1:
                    raise AcceptanceError("synthetic upload must produce five frames and a crop")
                with sessions() as db:
                    images = list(db.scalars(select(Image).order_by(Image.video_offset_seconds)))
                    crops = list(db.scalars(select(PersonCrop)))
                    if len(images) != 5 or not crops:
                        raise AcceptanceError(
                            "isolated upload did not persist its frames and crops"
                        )
                    source_url = images[0].source_video_url
                    if not source_url or not source_url.startswith("/data/videos/"):
                        raise AcceptanceError("isolated upload has no stored source video URL")
                    for index, image in enumerate(images):
                        offset = image.video_offset_seconds
                        if (
                            image.source_video_url != source_url
                            or not isinstance(offset, int | float)
                            or isinstance(offset, bool)
                            or not math.isfinite(offset)
                            or not math.isclose(offset, index / 10.0, abs_tol=0.005)
                        ):
                            raise AcceptanceError("synthetic frame media offsets are incorrect")
                        replay = client.get(f"/api/images/{image.id}/playback")
                        _validate_smoke_playback(
                            replay.status_code, replay.json(), source_url, offset
                        )
                    parents = {image.id: image for image in images}
                    for crop in crops:
                        parent = parents.get(crop.image_id)
                        if parent is None:
                            raise AcceptanceError(
                                "synthetic crop is not mapped to its parent frame"
                            )
                        replay = client.get(f"/api/person-crops/{crop.id}/playback")
                        _validate_smoke_playback(
                            replay.status_code,
                            replay.json(),
                            source_url,
                            parent.video_offset_seconds,
                        )
                ranged = client.get(source_url, headers={"Range": "bytes=0-15"})
                if (
                    ranged.status_code != 206
                    or ranged.content != video.read_bytes()[:16]
                    or not ranged.headers.get("content-range", "").startswith("bytes 0-15/")
                    or ranged.headers.get("accept-ranges") != "bytes"
                ):
                    raise AcceptanceError("stored synthetic video did not satisfy HTTP Range/206")
            return "isolated synthetic upload -> five frames/crops -> playback offsets -> HTTP 206"
        finally:
            api.dependency_overrides.clear()
            engine.dispose()


def _validate_smoke_playback(status: int, payload: Any, source: str, offset: float) -> None:
    """Check provenance from actual handlers without using real media identifiers."""
    data = _mapping(payload, "isolated playback")
    value = data.get("offset_seconds")
    if (
        status != 200
        or data.get("available") is not True
        or data.get("source_type") != "video_frame"
        or data.get("video_url") != source
        or not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not math.isclose(value, offset, abs_tol=0.005)
        or data.get("reason") is not None
    ):
        raise AcceptanceError("isolated playback did not preserve parent video/offset")


def model_smoke(settings: Settings) -> str:
    """Run local detector/face inference on generated pixels with downloads disabled."""
    import cv2
    import numpy as np

    from app.face_algorithms.insightface_cuda import InsightFaceCudaRecognizer, _insightface_app
    from app.services.frame_processing import create_person_detector

    detector_name = settings.person_detector.lower()
    if detector_name == "yolo":
        weights = Path(settings.yolo_model)
        if weights.suffix != ".pt" or not weights.is_file():
            raise AcceptanceError(
                "YOLO requires an existing local .pt checkpoint; downloads refused"
            )
    elif detector_name not in {"whole_frame", "hog"}:
        raise AcceptanceError("model smoke requires a local yolo/hog/whole_frame detector")
    with TemporaryDirectory(prefix="sightindex-model-smoke-") as directory:
        image = Path(directory) / "synthetic.png"
        if not cv2.imwrite(str(image), np.full((128, 128, 3), 96, dtype=np.uint8)):
            raise AcceptanceError("could not encode synthetic model input")
        create_person_detector(settings).detect(image)
        if settings.face_embedding_provider.lower() != "insightface":
            raise AcceptanceError("face model smoke requires the insightface provider")
        recognizer = InsightFaceCudaRecognizer(
            model_name=settings.face_insightface_model,
            det_size=settings.face_insightface_det_size,
            device=settings.face_embedding_device,
            root=settings.face_insightface_root,
            allow_download=False,
        )
        required = ("det_10g.onnx", "w600k_r50.onnx")
        if not all((recognizer.model_dir() / name).is_file() for name in required):
            raise AcceptanceError("face detector/recognizer assets are missing; downloads refused")
        info = recognizer.info()
        provider = (
            "CUDAExecutionProvider" if info.device.startswith("cuda") else "CPUExecutionProvider"
        )
        if provider not in info.available_providers:
            raise AcceptanceError("configured face execution provider is unavailable")
        recognizer.extract(image)
        from insightface.app import FaceAnalysis

        face_app = _insightface_app(
            recognizer.device,
            recognizer.model_name,
            recognizer.det_size,
            str(recognizer.root) if recognizer.root else None,
            FaceAnalysis,
            tuple(recognizer.providers),
        )
        for model in face_app.models.values():
            if model.session.get_providers()[0] != provider:
                raise AcceptanceError("face model execution fell back from the configured provider")
        embedding = face_app.models["recognition"].get_feat(
            np.full((112, 112, 3), 96, dtype=np.uint8)
        )
        values = np.asarray(embedding).reshape(-1)
        if values.size != settings.face_embedding_dim or not np.isfinite(values).all():
            raise AcceptanceError("face recognizer returned an invalid synthetic embedding")
    return (
        "local detector and face inference loaded; synthetic pixels do not establish model quality"
    )


def parse_stacks(value: str) -> frozenset[str]:
    """Parse the shared whitespace-separated stack selection."""
    selected = frozenset(value.split())
    if not selected or selected - STACKS:
        raise AcceptanceError("--stacks must contain only base, reid, embedding, semantic")
    return selected | {"base"}


def _add_application_path() -> None:
    """Support both repository paths and /opt/sightindex/deployment_verify.py."""
    location = Path(__file__).resolve()
    for candidate in (location.parent, location.parents[2]):
        if (candidate / "app" / "config" / "settings.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return


def main(argv: Sequence[str] | None = None) -> int:
    """Print bounded, credential-safe acceptance results and return failure status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stacks", default="base", help="base [reid] [embedding] [semantic]")
    parser.add_argument("--upload-smoke", action="store_true", help="isolated generated video only")
    parser.add_argument("--model-smoke", action="store_true", help="local assets, generated pixels")
    args = parser.parse_args(argv)
    _add_application_path()
    try:
        from app.config.settings import Settings

        stacks = parse_stacks(args.stacks)
        settings = Settings()
        results = verify_runtime(settings, stacks, application_http(settings, os.environ))
    except Exception as exc:
        detail = (
            str(exc)
            if isinstance(exc, AcceptanceError)
            else f"configuration failed ({type(exc).__name__})"
        )
        print(f"[FAIL] configuration: {detail}", file=sys.stderr)
        return 1
    if args.upload_smoke:
        results.append(_run_check("upload smoke", isolated_upload_smoke))
    if args.model_smoke:
        results.append(_run_check("model smoke", lambda: model_smoke(settings)))
    if settings.video_preparation_enabled:
        results.append(
            _run_check("video preparation tools", lambda: verify_video_preparation_tools(settings))
        )
    for result in results:
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.detail}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
