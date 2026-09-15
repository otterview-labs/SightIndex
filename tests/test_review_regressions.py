import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from test_reid import load_app


@pytest.fixture
def reviewed_app(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "review-regressions",
        MILVUS_ENABLED="false",
        REID_ENABLED="false",
        VECTOR_INDEX_ON_INGEST="false",
        VLM_PROVIDER="none",
        VLM_STRUCTURED_ON_INGEST="false",
        VLM_STRUCTURED_BACKGROUND="false",
        FACE_RECOGNITION_ON_INGEST="false",
        APPEARANCE_TONE_ON_INGEST="false",
        STREAM_AUTOSTART_RUNNING="false",
        APP_BASIC_AUTH_USERNAME="",
        APP_BASIC_AUTH_PASSWORD="",
    )
    main.init_db()
    return main


def _create_image(settings, db):
    from PIL import Image as PillowImage

    from app.models.media import Image

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    path = settings.uploads_dir / "review.png"
    PillowImage.new("RGB", (128, 256), (96, 96, 96)).save(path)
    image = Image(image_url="/data/uploads/review.png")
    db.add(image)
    db.commit()
    db.refresh(image)
    return image


@pytest.mark.parametrize("assignment", ["manual", "unlabel", "legacy", "face"])
def test_name_search_does_not_restore_a_rejected_historical_identity(reviewed_app, assignment):
    from app.db.session import SessionLocal
    from app.models.events import RecognitionEvent
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.schemas.media import VisualSearchRequest
    from app.services.persons import PersonService
    from app.services.search import VisualSearchService

    with SessionLocal() as db:
        old_person = Person(name="审计甲")
        new_person = Person(name="审计乙")
        image = Image(image_url="/data/review.png")
        db.add_all([old_person, new_person, image])
        db.flush()
        crop = PersonCrop(image_id=image.id, crop_url="/data/crop.png", bbox={})
        db.add(crop)
        db.flush()
        event = RecognitionEvent(
            image_id=image.id,
            crop_id=crop.id,
            person_id=old_person.id,
            result_type="known",
            recognized_at=datetime.now(UTC),
        )
        db.add(event)
        db.commit()
        service = PersonService(db, reviewed_app.get_settings())
        if assignment == "unlabel":
            service.unlabel_crop(crop)
        elif assignment == "manual":
            service.label_crop(new_person, crop)
        else:
            crop.person_id = new_person.id
            crop.person_id_source = None if assignment == "legacy" else "face"
            db.commit()
        result = VisualSearchService(db).search(VisualSearchRequest(query=old_person.name))
        assert result.items == []
        assert db.get(RecognitionEvent, event.id).person_id == old_person.id


def test_name_search_filters_rejected_events_before_top_k_and_keeps_legacy_recall(reviewed_app):
    from app.db.session import SessionLocal
    from app.models.events import RecognitionEvent
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person
    from app.schemas.media import VisualSearchRequest
    from app.services.search import VisualSearchService

    with SessionLocal() as db:
        person = Person(name="审计甲")
        image = Image(image_url="/data/review.png")
        db.add_all([person, image])
        db.flush()
        rejected = PersonCrop(
            image_id=image.id, crop_url="/data/rejected.png", bbox={}, person_id_source="manual"
        )
        legacy = PersonCrop(image_id=image.id, crop_url="/data/legacy.png", bbox={})
        db.add_all([rejected, legacy])
        db.flush()
        now = datetime.now(UTC)
        for crop, occurred in ((legacy, now - timedelta(hours=1)), (rejected, now)):
            db.add(RecognitionEvent(
                image_id=image.id, crop_id=crop.id, person_id=person.id,
                result_type="known", recognized_at=occurred,
            ))
        db.commit()
        result = VisualSearchService(db).search(
            VisualSearchRequest(query=person.name, top_k=1)
        )
        assert [item.crop_id for item in result.items] == [legacy.id]


@pytest.mark.parametrize("timestamps", ["distinct", "same", "server_default"])
def test_structured_search_reaches_matches_beyond_first_500_rows(reviewed_app, timestamps):
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.schemas.common import SearchFilters
    from app.services.search import StructuredSearchService

    camera_id = uuid.uuid4()
    with SessionLocal() as db:
        image = Image(image_url="/data/review.png")
        db.add(image)
        db.flush()
        first_time = datetime(2026, 1, 1, tzinfo=UTC)
        for index in range(1101):
            created_at = (
                first_time if timestamps == "same" else first_time + timedelta(seconds=index)
            )
            db.add(PersonCrop(
                id=uuid.UUID(int=index + 1),
                image_id=image.id,
                crop_url=f"/data/{index}.png",
                bbox={"label": "person"},
                camera_id=camera_id,
                attributes={"clothing": {"upper_color": "red" if index == 0 else "blue"}},
                **({} if timestamps == "server_default" else {"created_at": created_at}),
            ))
        db.add(PersonCrop(
            image_id=image.id, crop_url="/data/other-camera.png", bbox={},
            camera_id=uuid.uuid4(), attributes={"clothing": {"upper_color": "red"}},
        ))
        db.commit()
        result, _ = StructuredSearchService(db).search(
            "红衣的人", top_k=1, filters=SearchFilters(camera_id=camera_id)
        )
        assert [item.crop_id for item in result.items] == [uuid.UUID(int=1)]


def test_structured_pagination_terminates_without_repeating_server_timestamps(reviewed_app):
    from sqlalchemy import event

    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.search import StructuredSearchService

    with SessionLocal() as db:
        image = Image(image_url="/data/review.png")
        db.add(image)
        db.flush()
        crop = PersonCrop(
            image_id=image.id, crop_url="/data/red.png", bbox={},
            attributes={"clothing": {"upper_color": "red"}},
        )
        db.add(crop)
        db.commit()
        engine = db.get_bind()
        queries = 0

        def bounded_scan(_connection, _cursor, statement, _parameters, _context, _executemany):
            nonlocal queries
            if "FROM person_crops" in statement:
                queries += 1
                assert queries <= 6, "Pagination is repeating the last page"

        event.listen(engine, "before_cursor_execute", bounded_scan)
        try:
            red, _ = StructuredSearchService(db).search("红衣的人", top_k=8)
            assert [item.crop_id for item in red.items] == [crop.id]
            blue, _ = StructuredSearchService(db).search("蓝衣的人", top_k=8)
            assert blue.items == []
        finally:
            event.remove(engine, "before_cursor_execute", bounded_scan)


@pytest.mark.parametrize("empty", [False, True])
def test_image_processing_retry_does_not_repeat_detection_or_create_files(
    reviewed_app, empty
):
    from app.db.session import SessionLocal
    from app.services.frame_processing import Detection, FrameProcessingService

    class Detector:
        calls = 0

        def detect(self, _path):
            self.calls += 1
            return [] if empty else [
                Detection(bbox={"x": 0, "y": 0, "width": 128, "height": 256}, confidence=1.0)
            ]

    settings = reviewed_app.get_settings()
    with SessionLocal() as db:
        image = _create_image(settings, db)
        detector = Detector()
        processor = FrameProcessingService(db, settings, detector=detector)
        first = processor.process_image(image)
        files = set(settings.data_dir.rglob("*"))
        second = processor.process_image(image)
        assert {crop.id for crop in second} == {crop.id for crop in first}
        assert detector.calls == 1
        assert image.processed_at is not None
        assert set(settings.data_dir.rglob("*")) == files


def test_existing_crops_keep_manual_identity_without_reprocessing(reviewed_app):
    from app.db.session import SessionLocal
    from app.models.media import PersonCrop
    from app.models.persons import Person
    from app.services.frame_processing import FrameProcessingService

    class Detector:
        def detect(self, _path):
            pytest.fail("Legacy crops must be reused without invoking the detector")

    settings = reviewed_app.get_settings()
    with SessionLocal() as db:
        image = _create_image(settings, db)
        person = Person(name="人工确认")
        db.add(person)
        db.flush()
        crop = PersonCrop(
            image_id=image.id, crop_url="/data/legacy.png", bbox={},
            person_id=person.id, person_id_source="manual",
        )
        db.add(crop)
        db.commit()
        result = FrameProcessingService(db, settings, detector=Detector()).process_image(image)
        assert [item.id for item in result] == [crop.id]
        assert result[0].person_id == person.id
        assert result[0].person_id_source == "manual"
        assert image.processed_at is not None


def test_concurrent_processing_requests_share_one_committed_result(reviewed_app):
    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.frame_processing import Detection, FrameProcessingService

    class Detector:
        calls = 0

        def detect(self, _path):
            self.calls += 1
            return [Detection(
                bbox={"x": 0, "y": 0, "width": 128, "height": 256}, confidence=1.0
            )]

    settings = reviewed_app.get_settings()
    with SessionLocal() as db:
        image_id = _create_image(settings, db).id
    barrier = threading.Barrier(2)
    detector = Detector()

    def process():
        with SessionLocal() as db:
            image = db.get(Image, image_id)
            barrier.wait(timeout=5)
            crops = FrameProcessingService(db, settings, detector=detector).process_image(image)
            return {crop.id for crop in crops}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process) for _ in range(2)]
        results = [future.result(timeout=15) for future in futures]
    assert results[0] == results[1]
    assert detector.calls == 1
    with SessionLocal() as db:
        assert len(list(db.scalars(select(PersonCrop)))) == 1


def test_failed_processing_rolls_back_completion_and_can_retry(reviewed_app, monkeypatch):
    from app.db.session import SessionLocal
    from app.services.frame_processing import FrameProcessingService

    settings = reviewed_app.get_settings()
    with SessionLocal() as db:
        image = _create_image(settings, db)
        processor = FrameProcessingService(db, settings)
        files_before = set(path for path in settings.data_dir.rglob("*") if path.is_file())
        original_enqueue = processor._enqueue_index_jobs

        def fail(*_args):
            raise RuntimeError("simulated queue failure")

        monkeypatch.setattr(processor, "_enqueue_index_jobs", fail)
        with pytest.raises(RuntimeError, match="simulated queue failure"):
            processor.process_image(image)
        db.refresh(image)
        assert image.processed_at is None
        assert {path for path in settings.data_dir.rglob("*") if path.is_file()} == files_before
        monkeypatch.setattr(processor, "_enqueue_index_jobs", original_enqueue)
        assert len(processor.process_image(image)) == 1
        assert image.processed_at is not None


def test_processing_retry_keeps_the_original_durable_index_job(reviewed_app, monkeypatch):
    from sqlalchemy import select

    from app.db.session import SessionLocal
    from app.models.vectors import VectorIndexJob
    from app.services.frame_processing import FrameProcessingService
    from app.services.vector_index_queue import attribute_queue, vector_index_queue

    settings = reviewed_app.get_settings()
    settings.reid_enabled = True
    settings.reid_service_url = "http://unused.invalid"
    settings.milvus_enabled = True
    settings.reid_index_on_ingest = True
    monkeypatch.setattr(vector_index_queue, "wake", lambda *_args: None)
    monkeypatch.setattr(attribute_queue, "wake", lambda *_args: None)
    with SessionLocal() as db:
        image = _create_image(settings, db)
        processor = FrameProcessingService(db, settings)
        first = processor.process_image(image)
        job = db.scalars(select(VectorIndexJob)).one()
        original_job_id = job.id
        assert job.object_id == first[0].id
        assert job.target == "reid_person_crop"
        second = processor.process_image(image)
        assert [crop.id for crop in second] == [crop.id for crop in first]
        assert db.scalars(select(VectorIndexJob)).one().id == original_job_id


def test_processing_marker_migration_preserves_legacy_rows(reviewed_app, monkeypatch):
    from sqlalchemy import create_engine, inspect, text

    from app.db import session

    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE images (id VARCHAR PRIMARY KEY, image_url VARCHAR)"))
        connection.execute(text("INSERT INTO images VALUES ('legacy', '/data/legacy.png')"))
    monkeypatch.setattr(session, "engine", engine)
    try:
        session._ensure_compatible_schema()
        session._ensure_compatible_schema()
        columns = {column["name"] for column in inspect(engine).get_columns("images")}
        assert "processed_at" in columns
        with engine.connect() as connection:
            assert connection.execute(text("SELECT * FROM images")).one() == (
                "legacy", "/data/legacy.png", None
            )
    finally:
        engine.dispose()


def test_preview_releases_database_connection_before_streaming(reviewed_app, monkeypatch):
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    from app.db import session
    from app.models.media import VideoStream
    from app.services.media import MediaService

    engine = create_engine(
        reviewed_app.get_settings().database_url, pool_size=1, max_overflow=0, pool_timeout=0.1
    )
    monkeypatch.setattr(session, "SessionLocal", sessionmaker(bind=engine))
    with session.SessionLocal() as db:
        stream = VideoStream(name="simulation", stream_url="rtsp://unused.invalid")
        db.add(stream)
        db.commit()
        stream_id = stream.id

    def preview(*_args, **_kwargs):
        assert engine.pool.checkedout() == 0
        return iter([b"--frame\r\nsimulation\r\n"])

    monkeypatch.setattr(MediaService, "stream_mjpeg_frames", preview)
    app = reviewed_app.create_app()

    async def check():
        first_body = asyncio.Event()
        release = asyncio.Event()
        request_sent = False

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await release.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                first_body.set()
                await release.wait()

        path = f"/api/streams/{stream_id}/mjpeg"
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "method": "GET", "scheme": "http", "path": path, "raw_path": path.encode(),
            "query_string": b"", "headers": [], "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80), "http_version": "1.1", "root_path": "",
        }
        task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(first_body.wait(), timeout=5)
            assert engine.pool.checkedout() == 0
            with engine.connect() as connection:
                assert connection.execute(text("SELECT 1")).scalar() == 1
        finally:
            release.set()
            await asyncio.wait_for(task, timeout=5)

    try:
        asyncio.run(check())
    finally:
        engine.dispose()
