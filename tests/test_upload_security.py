import asyncio
from io import BytesIO

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin
from test_reid import load_app


@pytest.fixture
def upload_app(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch, tmp_path, "upload-security",
        MILVUS_ENABLED="false",
        REID_ENABLED="false",
        VECTOR_INDEX_ON_INGEST="false",
        FACE_RECOGNITION_ON_INGEST="false",
        STREAM_AUTOSTART_RUNNING="false",
        APP_BASIC_AUTH_USERNAME="",
        APP_BASIC_AUTH_PASSWORD="",
    )
    main.init_db()
    return main


def _image_bytes(image_format: str = "PNG", size: tuple[int, int] = (32, 64)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (96, 80, 70)).save(buffer, format=image_format)
    return buffer.getvalue()


def _assert_no_uploads(main) -> None:
    from sqlalchemy import func, select

    from app.db.session import SessionLocal
    from app.models.media import Image as MediaImage

    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(MediaImage)) == 0
    settings = main.get_settings()
    assert not list(settings.uploads_dir.iterdir())
    assert not list(settings.videos_dir.iterdir())


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP", "BMP", "GIF"])
def test_valid_images_are_normalized_before_publication(upload_app, image_format):
    client = TestClient(upload_app.create_app())
    original = _image_bytes(image_format) + b"<script>appended-content</script>"
    response = client.post(
        "/api/images/upload",
        files={"file": (f"frame.{image_format.lower()}", original, "application/octet-stream")},
    )
    assert response.status_code == 200
    image_url = response.json()["image_url"]
    assert image_url.endswith(".png")
    served = client.get(image_url)
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"
    assert served.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in served.headers["content-security-policy"]
    assert b"appended-content" not in served.content
    with Image.open(BytesIO(served.content)) as normalized:
        normalized.load()
        assert normalized.size == (32, 64)
        assert normalized.info == {}


def test_normalization_preserves_orientation_and_removes_metadata(upload_app):
    from app.services.storage import StorageService

    image = Image.new("RGB", (16, 32), (96, 80, 70))
    exif = image.getexif()
    exif[274] = 6
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Description", "<script>metadata-content</script>")
    buffer = BytesIO()
    image.save(buffer, format="PNG", exif=exif, pnginfo=metadata)
    buffer.seek(0)
    settings = upload_app.get_settings()
    url = StorageService(settings).save_upload(UploadFile(buffer, filename="frame.png"))
    path = settings.data_dir / url.removeprefix("/data/")
    assert b"metadata-content" not in path.read_bytes()
    with Image.open(path) as normalized:
        assert normalized.size == (32, 16)
        assert normalized.info == {}
        assert not normalized.getexif()
        assert normalized.getpixel((0, 0)) == (96, 80, 70)


@pytest.mark.parametrize(
    ("filename", "content", "status"),
    [
        ("page.html", b"<!doctype html><h1>not an image</h1>", 415),
        ("vector.svg", b"<svg xmlns='http://www.w3.org/2000/svg'/>", 415),
        ("fake.jpg", b"<!doctype html><h1>not an image</h1>", 400),
        ("broken.png", b"\x89PNG\r\n\x1a\nbroken", 400),
        ("empty.jpg", b"", 400),
    ],
)
def test_invalid_images_leave_no_public_file_or_database_row(
    upload_app, filename, content, status
):
    response = TestClient(upload_app.create_app()).post(
        "/api/images/upload", files={"file": (filename, content, "image/jpeg")}
    )
    assert response.status_code == status
    _assert_no_uploads(upload_app)


@pytest.mark.parametrize("limit_kind", ["bytes", "pixels", "normalized"])
def test_image_limits_return_413_without_publishing(upload_app, limit_kind):
    settings = upload_app.get_settings()
    content = _image_bytes()
    if limit_kind == "bytes":
        settings.upload_image_max_bytes = len(content) - 1
    elif limit_kind == "pixels":
        settings.upload_image_max_pixels = 32 * 64 - 1
    else:
        content = _image_bytes("WEBP", (512, 512))
        settings.upload_image_max_bytes = len(content)
    response = TestClient(upload_app.create_app()).post(
        "/api/images/upload", files={"file": ("frame.png", content, "image/png")}
    )
    assert response.status_code == 413
    _assert_no_uploads(upload_app)


@pytest.mark.parametrize("filename", ["old.html", "old.svg", ".env", "metadata.json", "model.onnx"])
def test_legacy_nonmedia_files_are_not_publicly_served(upload_app, filename):
    settings = upload_app.get_settings()
    (settings.uploads_dir / filename).write_text("legacy content", encoding="utf-8")
    response = TestClient(upload_app.create_app()).get(f"/data/uploads/{filename}")
    assert response.status_code == 404


def test_legacy_media_stays_accessible_but_requires_configured_auth(upload_app):
    settings = upload_app.get_settings()
    settings.app_basic_auth_username = "test-viewer"
    settings.app_basic_auth_password = "test-password"
    (settings.uploads_dir / "legacy.jpg").write_bytes(_image_bytes("JPEG"))
    client = TestClient(upload_app.create_app())
    assert client.get("/data/uploads/legacy.jpg").status_code == 401
    response = client.get(
        "/data/uploads/legacy.jpg", auth=("test-viewer", "test-password")
    )
    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in response.headers["content-security-policy"]


@pytest.mark.parametrize(
    ("filename", "content", "status"),
    [("fake.html", b"<!doctype html>", 415), ("large.mp4", b"x" * 1025, 413)],
)
def test_rejected_video_uploads_leave_no_partial_file(upload_app, filename, content, status):
    upload_app.get_settings().upload_video_max_bytes = 1024
    response = TestClient(upload_app.create_app()).post(
        "/api/videos/upload", files={"file": (filename, content, "video/mp4")}
    )
    assert response.status_code == status
    _assert_no_uploads(upload_app)


def test_interrupted_write_removes_only_its_own_partial_file(upload_app, tmp_path):
    from app.services.storage import StorageService

    class InterruptedSource(BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError("simulated source interruption")
            return super().read(size)

    target = tmp_path / "partial.mp4"
    with pytest.raises(OSError, match="source interruption"):
        StorageService._save_limited(InterruptedSource(b"x" * 70000), target, 100000)
    assert not target.exists()
    target.write_bytes(b"existing file")
    with pytest.raises(FileExistsError):
        StorageService._save_limited(BytesIO(b"new file"), target, 100)
    assert target.read_bytes() == b"existing file"


def test_storage_failure_is_not_misreported_as_invalid_image(upload_app, monkeypatch):
    from app.services.storage import StorageService

    def fail(*_args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(StorageService, "_save_limited", fail)
    with pytest.raises(OSError, match="disk failure"):
        StorageService(upload_app.get_settings()).save_upload(
            UploadFile(BytesIO(_image_bytes()), filename="frame.png")
        )


def _scope(path: str, headers: list[tuple[bytes, bytes]]) -> dict:
    return {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
        "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "headers": headers, "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80), "http_version": "1.1", "root_path": "",
    }


def test_declared_oversize_is_rejected_without_reading_body(upload_app):
    from app.api.upload_limits import UploadSizeLimitMiddleware

    async def unexpected(*_args):
        pytest.fail("Oversize Content-Length must be rejected before reading or parsing")

    messages = []

    async def send(message):
        messages.append(message)

    middleware = UploadSizeLimitMiddleware(unexpected, upload_app.get_settings())
    asyncio.run(middleware(
        _scope("/api/images/upload", [(b"content-length", b"999999999")]), unexpected, send
    ))
    assert messages[0]["status"] == 413


@pytest.mark.parametrize("declared_length", [None, b"1"])
def test_chunked_or_underreported_upload_is_bounded_and_closes_spooled_files(
    upload_app, monkeypatch, declared_length
):
    import starlette.formparsers as parsers

    settings = upload_app.get_settings()
    settings.upload_image_max_bytes = 1024
    created = []
    original_spool = parsers.SpooledTemporaryFile

    def tracked_spool(*args, **kwargs):
        file = original_spool(*args, **kwargs)
        created.append(file)
        return file

    monkeypatch.setattr(parsers, "SpooledTemporaryFile", tracked_spool)
    header = (
        b"--test-boundary\r\n"
        b'Content-Disposition: form-data; name="file"; filename="large.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n"
    )
    chunks = iter([header + b"x" * 1000, b"x" * 70000, b"\r\n--test-boundary--\r\n"])
    headers = [(b"content-type", b"multipart/form-data; boundary=test-boundary")]
    if declared_length is not None:
        headers.append((b"content-length", declared_length))
    messages = []

    async def receive():
        chunk = next(chunks, None)
        return {"type": "http.request", "body": chunk or b"", "more_body": chunk is not None}

    async def send(message):
        messages.append(message)

    asyncio.run(upload_app.create_app()(
        _scope("/api/images/upload", headers), receive, send
    ))
    assert [message["status"] for message in messages if "status" in message] == [413]
    assert created and all(file.closed for file in created)
    _assert_no_uploads(upload_app)


@pytest.mark.parametrize(
    ("path", "status"),
    [("/api/images/upload", 413), ("/api/videos/upload", 200), ("/api/videos/upload/", 200)],
)
def test_request_limit_uses_separate_video_budget(upload_app, path, status):
    from starlette.responses import JSONResponse

    from app.api.upload_limits import UploadSizeLimitMiddleware

    settings = upload_app.get_settings()
    settings.upload_image_max_bytes = 1024
    settings.upload_video_max_bytes = 128 * 1024
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"x" * 80000, "more_body": False}

    async def send(message):
        messages.append(message)

    async def consume(scope, receive, send):
        await receive()
        await JSONResponse({"ok": True})(scope, receive, send)

    asyncio.run(UploadSizeLimitMiddleware(consume, settings)(_scope(path, []), receive, send))
    assert messages[0]["status"] == status
