"""Stored-video playback tests use synthetic paths/pixels and isolated SQLite only."""

from __future__ import annotations

import importlib
import sys
import uuid
from base64 import b64encode
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import sessionmaker


@pytest.mark.parametrize("offset", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_playback_response_refuses_negative_or_nonfinite_offsets(offset: float) -> None:
    from app.schemas.media import VideoPlaybackRead

    with pytest.raises(ValidationError):
        VideoPlaybackRead(available=True, source_type="video_frame", offset_seconds=offset)


def _settings(tmp_path: Path) -> Any:
    from app.config.settings import Settings

    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        person_detector="whole_frame",
        vector_index_on_ingest=False,
        reid_enabled=False,
        milvus_enabled=False,
        face_recognition_on_ingest=False,
    )


def _image(**values: Any) -> Any:
    from app.models.media import Image

    return Image(
        image_url="/data/frames/example.jpg",
        source_type=values.pop("source_type", "video_frame"),
        **values,
    )


def _source(settings: Any, name: str = "example.mp4") -> Path:
    settings.videos_dir.mkdir(parents=True, exist_ok=True)
    source = settings.videos_dir / name
    source.write_bytes(b"synthetic stored media")
    return source


@pytest.mark.parametrize("offset", [0.0, 12.375])
def test_verified_media_position_preserves_zero_and_capture_time(
    tmp_path: Path, offset: float
) -> None:
    from app.services.video_playback import video_playback_for_image

    settings = _settings(tmp_path)
    _source(settings)
    stamp = datetime(2026, 10, 6, tzinfo=UTC)
    result = video_playback_for_image(
        _image(
            source_video_url="/data/videos/example.mp4",
            video_offset_seconds=offset,
            captured_at=stamp,
        ),
        settings,
    )

    assert result.available
    assert result.video_url == "/data/videos/example.mp4"
    assert result.offset_seconds == offset
    assert result.captured_at == stamp
    assert result.reason is None


@pytest.mark.parametrize(
    ("source_type", "reason"),
    [
        ("upload", "not_video"),
        ("dataset", "not_video"),
        ("stream_frame", "recording_not_configured"),
        ("stream", "recording_not_configured"),
        ("video_frame", "source_missing"),
    ],
)
def test_unknown_sources_are_not_inferred_from_filenames_or_capture_dates(
    tmp_path: Path, source_type: str, reason: str
) -> None:
    from app.services.video_playback import video_playback_for_image

    settings = _settings(tmp_path)
    _source(settings)
    result = video_playback_for_image(_image(source_type=source_type), settings)

    assert not result.available
    assert result.reason == reason
    assert result.video_url is None
    assert result.offset_seconds is None


@pytest.mark.parametrize("offset", [None, float("nan"), float("inf"), -1.0, True, "invalid"])
def test_invalid_offsets_are_not_serialized_or_silently_changed_to_zero(
    tmp_path: Path, offset: float | None
) -> None:
    from app.services.video_playback import video_playback_for_image

    settings = _settings(tmp_path)
    _source(settings)
    result = video_playback_for_image(
        _image(source_video_url="/data/videos/example.mp4", video_offset_seconds=offset), settings
    )

    assert result.model_dump_json()
    assert not result.available
    assert result.reason == "offset_unknown"
    assert result.offset_seconds is None


@pytest.mark.parametrize(
    "url",
    [
        "https://example.test/movie.mp4",
        "rtsp://camera.test/live",
        "/data/uploads/example.mp4",
        "/data/videos/../example.mp4",
        "/data/videos/sub/example.mp4",
        "/data/videos/%2e%2e/example.mp4",
        "/data/videos/example.mp4?download=1",
        "/data/videos/example.mp4#fragment",
        "/data/videos/example.html",
        "/data/videos/..\\example.mp4",
    ],
)
def test_unsafe_or_nonvideo_sources_are_never_exposed(tmp_path: Path, url: str) -> None:
    from app.services.video_playback import video_playback_for_image

    settings = _settings(tmp_path)
    _source(settings)
    result = video_playback_for_image(
        _image(source_video_url=url, video_offset_seconds=0), settings
    )

    assert not result.available
    assert result.reason == "invalid_source"
    assert result.video_url is None


def test_missing_source_returns_explanation(tmp_path: Path) -> None:
    from app.services.video_playback import video_playback_for_image

    result = video_playback_for_image(
        _image(source_video_url="/data/videos/missing.mp4", video_offset_seconds=0),
        _settings(tmp_path),
    )
    assert not result.available
    assert result.reason == "media_missing"


@pytest.mark.parametrize("inside", [False, True])
def test_symlink_source_is_rejected_even_if_target_is_inside(tmp_path: Path, inside: bool) -> None:
    from app.services.video_playback import video_playback_for_image

    settings = _settings(tmp_path)
    target = _source(settings) if inside else tmp_path / "outside.mp4"
    if not inside:
        settings.videos_dir.mkdir(parents=True)
        target.write_bytes(b"outside fixture")
    (settings.videos_dir / "linked.mp4").symlink_to(target)
    result = video_playback_for_image(
        _image(source_video_url="/data/videos/linked.mp4", video_offset_seconds=0), settings
    )
    assert not result.available
    assert result.reason == "invalid_source"


def test_directory_and_symlink_video_directory_are_rejected(tmp_path: Path) -> None:
    from app.services.video_playback import video_playback_for_image

    settings = _settings(tmp_path)
    settings.videos_dir.mkdir(parents=True)
    (settings.videos_dir / "directory.mp4").mkdir()
    result = video_playback_for_image(
        _image(source_video_url="/data/videos/directory.mp4", video_offset_seconds=0), settings
    )
    assert result.reason == "invalid_source"
    (settings.videos_dir / "directory.mp4").rmdir()
    settings.videos_dir.rmdir()
    target = tmp_path / "external-videos"
    target.mkdir()
    (target / "example.mp4").write_bytes(b"synthetic fixture")
    settings.videos_dir.symlink_to(target, target_is_directory=True)
    result = video_playback_for_image(
        _image(source_video_url="/data/videos/example.mp4", video_offset_seconds=0), settings
    )
    assert result.reason == "invalid_source"


def test_additive_migration_is_idempotent_and_preserves_existing_protection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.db import session

    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.sqlite'}")
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE images (id VARCHAR PRIMARY KEY, processed_at DATETIME)")
        )
        connection.execute(
            text(
                "INSERT INTO images (id, processed_at) VALUES ('old-image', '2026-10-06 10:00:00')"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE person_crops (id VARCHAR PRIMARY KEY, person_id VARCHAR, "
                "person_id_source VARCHAR, attributes JSON, "
                "created_at DATETIME, captured_at DATETIME)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO person_crops (id, person_id, person_id_source) "
                "VALUES ('protected-crop', 'manual-person', 'manual')"
            )
        )
    monkeypatch.setattr(session, "engine", engine)
    try:
        session._ensure_compatible_schema()
        session._ensure_compatible_schema()
        columns = {column["name"] for column in inspect(engine).get_columns("images")}
        assert {"processed_at", "source_video_url", "video_offset_seconds"} <= columns
        with engine.connect() as connection:
            assert connection.execute(text("SELECT * FROM images")).one() == (
                "old-image",
                "2026-10-06 10:00:00",
                None,
                None,
            )
            assert connection.execute(
                text("SELECT person_id, person_id_source FROM person_crops")
            ).one() == ("manual-person", "manual")
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("position", "frame_index", "fps", "expected"),
    [
        (0, 0, 25, 0),
        (1250, 25, 25, 1.25),
        (1250, 25, float("nan"), 1.25),
        (1250, 25, float("inf"), 1.25),
        (0, 25, 25, None),
        (float("nan"), 25, 25, None),
        (float("inf"), 25, 25, None),
        (-1000, 25, 25, None),
        (None, 25, 25, None),
        (True, 25, 25, None),
        ("invalid", 25, 25, None),
        (1250, -1, 25, None),
        (None, 25, 0, None),
        (0, 25, 0, None),
        (None, 25, float("nan"), None),
        (None, 25, float("inf"), None),
    ],
)
def test_same_frame_media_offset_never_guesses_from_fps(
    position: object, frame_index: int, fps: float, expected: float | None
) -> None:
    from app.services.video_processing import VideoProcessingService

    capture = SimpleNamespace(get=lambda _prop: position)
    cv2 = SimpleNamespace(CAP_PROP_POS_MSEC=1)
    assert (
        VideoProcessingService._frame_offset_seconds(capture, cv2, frame_index=frame_index, fps=fps)
        == expected
    )


@pytest.mark.parametrize(
    ("position", "high_water", "expected"),
    [
        (1250.0, 1250.0, None),
        (1000.0, 1250.0, None),
        (1100.0, 1250.0, None),
        (1250.001, 1250.0, 1.250001),
        (2000.0, 1250.0, 2.0),
        (float("nan"), 1250.0, None),
        (float("inf"), 1250.0, None),
        (-10.0, 1250.0, None),
    ],
)
def test_recovered_media_clock_must_exceed_the_high_water_mark(
    position: float, high_water: float, expected: float | None
) -> None:
    from app.services.video_processing import VideoProcessingService

    actual = VideoProcessingService._offset_from_media_position(
        position, frame_index=25, fps=25.0, previous_position_ms=high_water
    )
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)


def _sessions(tmp_path: Path) -> tuple[Any, Any]:
    from app.db.session import Base

    for name in ("persons", "events", "media", "vectors", "chat", "reid"):
        importlib.import_module(f"app.models.{name}")
    engine = create_engine(f"sqlite:///{tmp_path / 'media.sqlite'}")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


@pytest.mark.parametrize(
    ("positions", "fps", "expected"),
    [
        ([0.0, 1250.0], 0.0, [0.0, 1.25]),
        ([float("nan"), float("inf")], 0.0, [None, None]),
        pytest.param([1000.0, 1000.0, 1000.0], 20.0, [1.0, None, None], id="stalled-fps"),
        pytest.param([1000.0, 1000.0, 1000.0], 0.0, [1.0, None, None], id="stalled-unknown"),
        pytest.param([1250.0, 1000.0, 1100.0], 20.0, [1.25, None, None], id="backward-fps"),
        pytest.param([1250.0, 1000.0, 1100.0], 0.0, [1.25, None, None], id="backward-unknown"),
        pytest.param(
            [1250.0, 1000.0, 1100.0, 1250.0, 1500.0, 1400.0, 1600.0],
            20.0,
            [1.25, None, None, None, 1.5, None, 1.6],
            id="clock-recovers-past-high-water",
        ),
        pytest.param([0.0, 0.0, 0.0], 20.0, [0.0, None, None], id="unsupported-zero-clock"),
        pytest.param(
            [0.0, 200.0, 100.0, 250.0, 300.0],
            20.0,
            [0.0, 0.2, None, 0.25, 0.3],
            id="omitted-frame-keeps-high-water",
        ),
        pytest.param(
            [0.0, float("nan"), 80.0, float("inf"), 90.0],
            20.0,
            [0.0, None, 0.08, None, 0.09],
            id="invalid-clock-recovers",
        ),
        pytest.param([0.0, 80.0, 220.0], float("nan"), [0.0, 0.08, 0.22], id="nan-fps"),
        pytest.param([0.0, 80.0, 220.0], float("inf"), [0.0, 0.08, 0.22], id="infinite-fps"),
        pytest.param([0.0, 80.0, 220.0], -20.0, [0.0, 0.08, 0.22], id="negative-fps"),
        pytest.param([0.0, 80.0, 220.0], 20.0, [0.0, 0.08, 0.22], id="valid-vfr"),
    ],
)
@pytest.mark.parametrize(
    ("use_upload", "frame_interval_seconds"),
    [(False, 0.01), (True, 0.01), (True, 0.1)],
    ids=["stored-path", "uploaded-file", "uploaded-sampled-frames"],
)
def test_video_processing_persists_source_and_offset_for_the_decoded_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    positions: list[float],
    fps: float,
    expected: list[float | None],
    use_upload: bool,
    frame_interval_seconds: float,
) -> None:
    from app.models.media import Image
    from app.services.time_utils import database_datetime
    from app.services.video_playback import video_playback_for_image
    from app.services.video_processing import VideoFrameFile, VideoProcessingService

    class Capture:
        index = -1
        released = False
        position_reads: list[int]

        def __init__(self) -> None:
            self.position_reads = []

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, Any]:
            self.index += 1
            return self.index < len(positions), SimpleNamespace(shape=(32, 32, 3), index=self.index)

        def get(self, prop: int) -> float:
            if prop == 1:
                return fps
            self.position_reads.append(self.index)
            return positions[self.index]

        def release(self) -> None:
            self.released = True

    capture = Capture()
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(VideoCapture=lambda _path: capture, CAP_PROP_FPS=1, CAP_PROP_POS_MSEC=2),
    )
    engine, sessions = _sessions(tmp_path)
    settings = _settings(tmp_path)
    source = _source(settings)
    stamp = datetime(2026, 10, 6, tzinfo=UTC)
    try:
        with sessions() as db:
            service = VideoProcessingService(db, settings)
            service.processor = SimpleNamespace(
                detect_image_path=lambda _path: [],
                quality_filter_detections=lambda items, _width, _height: items,
                process_image=lambda _image, detections: [],
            )

            def write_frame(*, frame: Any, captured_at: datetime, **_kwargs: Any) -> VideoFrameFile:
                path = tmp_path / f"frame-{frame.index}.jpg"
                path.write_bytes(b"synthetic pixels")
                return VideoFrameFile(
                    url=f"/data/frames/frame-{frame.index}.jpg", path=path, captured_at=captured_at
                )

            monkeypatch.setattr(service, "_write_frame_file", write_frame)
            options = {
                "frame_interval_seconds": frame_interval_seconds,
                "max_frames": len(positions),
                "store_empty_frames": True,
                "captured_at": stamp,
            }
            if use_upload:
                result = service.process_upload(
                    UploadFile(file=BytesIO(b"synthetic video fixture"), filename="upload.mp4"),
                    **options,
                )
            else:
                result = service.process_video_path(source, "/data/videos/example.mp4", **options)
            images = [db.get(Image, image_id) for image_id in result.image_ids]
            # For the 20-FPS fixture, the longer interval omits odd frames. Their clocks must
            # still influence the following stored frame's timestamp validation.
            stored_offsets = (
                expected[::2] if frame_interval_seconds == 0.1 and fps == 20 else expected
            )
            assert result.frames_read == len(positions)
            assert result.frames_sampled == result.images_created == len(stored_offsets)
            assert all(image.source_video_url == result.video_url for image in images)
            assert [image.video_offset_seconds for image in images] == stored_offsets
            base_time = database_datetime(stamp, settings, "sqlite")
            for image, offset in zip(images, stored_offsets, strict=True):
                assert image.captured_at == base_time + timedelta(seconds=offset or 0.0)
                playback = video_playback_for_image(image, settings)
                assert playback.available == (offset is not None)
                assert playback.offset_seconds == offset
                assert playback.reason == ("offset_unknown" if offset is None else None)
            if use_upload:
                assert service._resolve_data_url(result.video_url).read_bytes() == (
                    b"synthetic video fixture"
                )
    finally:
        engine.dispose()
    assert capture.released
    assert capture.position_reads == list(range(len(positions)))


def test_real_synthetic_fixed_rate_video_decodes_with_correct_stored_offsets(
    tmp_path: Path,
) -> None:
    """Verify actual local decoding without opening or downloading any surveillance media."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from app.models.media import Image
    from app.services.video_processing import VideoProcessingService

    settings = _settings(tmp_path)
    settings.videos_dir.mkdir(parents=True)
    source = settings.videos_dir / "fixed-rate.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("Local OpenCV has no MJPG encoder")
    try:
        for index in range(5):
            writer.write(np.full((48, 64, 3), 40 + index * 30, dtype=np.uint8))
    finally:
        writer.release()
    engine, sessions = _sessions(tmp_path)
    try:
        with sessions() as db:
            service = VideoProcessingService(db, settings)
            service.processor = SimpleNamespace(
                detect_image_path=lambda _path: [],
                quality_filter_detections=lambda items, _width, _height: items,
                process_image=lambda _image, detections: [],
            )
            result = service.process_video_path(
                source,
                "/data/videos/fixed-rate.avi",
                frame_interval_seconds=0.1,
                max_frames=5,
                store_empty_frames=True,
                captured_at=datetime(2026, 10, 6, tzinfo=UTC),
            )
            images = [db.get(Image, image_id) for image_id in result.image_ids]
            assert result.images_created == 5
            assert [image.video_offset_seconds for image in images] == pytest.approx(
                [0.0, 0.1, 0.2, 0.3, 0.4], abs=0.005
            )
            assert all(image.source_video_url == "/data/videos/fixed-rate.avi" for image in images)
            assert all(
                cv2.imread(str(settings.frames_dir / Path(image.image_url).name)) is not None
                for image in images
            )
    finally:
        engine.dispose()


def test_repeated_or_unknown_decoder_timestamps_do_not_overwrite_pixels(tmp_path: Path) -> None:
    from app.services.video_processing import VideoProcessingService

    settings = _settings(tmp_path)
    service = VideoProcessingService.__new__(VideoProcessingService)
    service.settings = settings

    def write_pixels(filename: str, pixels: bytes, _options: list[int]) -> bool:
        Path(filename).write_bytes(pixels)
        return True

    cv2 = SimpleNamespace(imwrite=write_pixels, IMWRITE_JPEG_QUALITY=1)
    stamp = datetime(2026, 10, 6, tzinfo=UTC)
    frames = [
        service._write_frame_file(
            video_path=tmp_path / "source.mp4", frame=pixels, cv2=cv2, captured_at=stamp
        )
        for pixels in (b"first decoded frame", b"second decoded frame")
    ]
    assert frames[0].path != frames[1].path
    assert frames[0].path.read_bytes() == b"first decoded frame"
    assert frames[1].path.read_bytes() == b"second decoded frame"


@pytest.mark.parametrize("committed", [False, True])
def test_upload_failure_only_removes_source_when_no_committed_frame_uses_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, committed: bool
) -> None:
    from app.services.video_processing import VideoProcessingService

    engine, sessions = _sessions(tmp_path)
    settings = _settings(tmp_path)
    try:
        with sessions() as db:
            service = VideoProcessingService(db, settings)

            def fail_processing(**kwargs: Any) -> None:
                if committed:
                    db.add(_image(source_video_url=kwargs["video_url"], video_offset_seconds=0))
                    db.commit()
                raise ValueError("synthetic processing failure")

            monkeypatch.setattr(service, "process_video_path", fail_processing)
            with pytest.raises(ValueError, match="synthetic processing failure"):
                service.process_upload(UploadFile(file=BytesIO(b"video fixture"), filename="x.mp4"))
            assert len(list(settings.videos_dir.iterdir())) == int(committed)
    finally:
        engine.dispose()


def test_upload_cleanup_database_failure_preserves_source_and_original_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.services.video_processing import VideoProcessingService

    engine, sessions = _sessions(tmp_path)
    settings = _settings(tmp_path)
    try:
        with sessions() as db:
            service = VideoProcessingService(db, settings)

            def processing_failure(**_kwargs: Any) -> None:
                raise ValueError("original processing failure")

            def database_failure(*_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("synthetic database outage")

            monkeypatch.setattr(service, "process_video_path", processing_failure)
            monkeypatch.setattr(db, "scalar", database_failure)
            with pytest.raises(ValueError, match="original processing failure"):
                service.process_upload(UploadFile(file=BytesIO(b"video fixture"), filename="x.mp4"))
            assert len(list(settings.videos_dir.iterdir())) == 1
    finally:
        engine.dispose()


def test_playback_endpoints_auth_parent_association_missing_ids_and_media_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'api.sqlite'}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PERSON_DETECTOR", "whole_frame")
    monkeypatch.setenv("VECTOR_INDEX_ON_INGEST", "false")
    monkeypatch.setenv("MILVUS_ENABLED", "false")
    monkeypatch.setenv("REID_ENABLED", "false")
    monkeypatch.setenv("APP_BASIC_AUTH_USERNAME", "viewer")
    monkeypatch.setenv("APP_BASIC_AUTH_PASSWORD", "fixture-only-password")
    for module_name in list(sys.modules):
        if module_name == "main" or module_name.startswith("app."):
            sys.modules.pop(module_name)
    main = importlib.import_module("main")
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop

    settings = main.get_settings()
    _source(settings)
    auth = {"Authorization": "Basic " + b64encode(b"viewer:fixture-only-password").decode("ascii")}
    with TestClient(main.create_app()) as client:
        with SessionLocal() as db:
            first = _image(source_video_url="/data/videos/example.mp4", video_offset_seconds=0)
            unrelated = _image(source_video_url="/data/videos/example.mp4", video_offset_seconds=99)
            stream = _image(source_type="stream_frame")
            legacy = _image()
            uploaded = _image(source_type="upload")
            db.add_all([first, unrelated, stream, legacy, uploaded])
            db.flush()
            crop = PersonCrop(image_id=first.id, crop_url="/data/crops/a.jpg", bbox={})
            db.add(crop)
            db.commit()
            first_id, crop_id = first.id, crop.id
            cases = [
                (stream.id, "recording_not_configured"),
                (legacy.id, "source_missing"),
                (uploaded.id, "not_video"),
            ]

        for route in (f"/api/images/{first_id}/playback", f"/api/person-crops/{crop_id}/playback"):
            assert client.get(route).status_code == 401
            response = client.get(route, headers=auth)
            assert response.status_code == 200
            assert response.json()["available"]
            assert response.json()["offset_seconds"] == 0
            assert response.json()["video_url"] == "/data/videos/example.mp4"
        for image_id, reason in cases:
            response = client.get(f"/api/images/{image_id}/playback", headers=auth)
            assert response.status_code == 200
            assert response.json()["reason"] == reason
        for route in ("images", "person-crops"):
            assert (
                client.get(f"/api/{route}/{uuid.uuid4()}/playback", headers=auth).status_code == 404
            )
        assert client.get("/data/videos/example.mp4").status_code == 401
        response = client.get("/data/videos/example.mp4", headers={**auth, "Range": "bytes=0-8"})
        assert response.status_code == 206
        assert response.content == b"synthetic"
        head = client.head("/data/videos/example.mp4", headers=auth)
        assert head.status_code == 200
        assert head.content == b""
        assert head.headers["accept-ranges"] == "bytes"
        beyond_end = client.get(
            "/data/videos/example.mp4", headers={**auth, "Range": "bytes=9999-"}
        )
        assert beyond_end.status_code == 416
        assert beyond_end.headers["content-range"] == "bytes */22"

        # Orphaned legacy crops never borrow an unrelated frame/video.
        with SessionLocal() as db:
            orphan = PersonCrop(image_id=uuid.uuid4(), crop_url="/data/crops/orphan.jpg", bbox={})
            db.add(orphan)
            db.commit()
            orphan_id = orphan.id
        assert (
            client.get(f"/api/person-crops/{orphan_id}/playback", headers=auth).status_code == 404
        )
        settings.videos_dir.joinpath("example.mp4").unlink()
        response = client.get(f"/api/person-crops/{crop_id}/playback", headers=auth)
        assert response.json()["reason"] == "media_missing"

    with SessionLocal() as db:
        assert db.scalar(select(Image.source_video_url).where(Image.id == first_id)) is not None
