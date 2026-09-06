from datetime import UTC, datetime, timedelta

from test_vector_index_queue_durability import load_app


def test_capture_health_is_based_on_read_heartbeat_not_model_updates(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "stream-health")
    from app.db.session import SessionLocal
    from app.models.media import VideoStream
    from app.schemas.media import VideoStreamRead

    main.init_db()
    with SessionLocal() as db:
        stream = VideoStream(name="test", stream_url="rtsp://x", status="running")
        db.add(stream)
        db.commit()
        db.refresh(stream)
        assert VideoStreamRead.model_validate(stream).capture_health == "unverified"
        stream.last_frame_read_at = datetime.now(UTC)
        assert VideoStreamRead.model_validate(stream).capture_health == "healthy"
        stream.last_frame_read_at = datetime.now(UTC) - timedelta(minutes=5)
        stream.updated_at = datetime.now(UTC)
        assert VideoStreamRead.model_validate(stream).capture_health == "stalled"
        stream.status = "stopped"
        assert VideoStreamRead.model_validate(stream).capture_health == "stopped"
        stream.status = "starting"
        assert VideoStreamRead.model_validate(stream).capture_health == "unverified"


def test_shutdown_retains_autostart_only_for_previously_active_cameras(monkeypatch, tmp_path):
    import threading

    main = load_app(monkeypatch, tmp_path, "stream-shutdown")
    from app.db.session import SessionLocal
    from app.models.media import VideoStream
    from app.services.stream_runtime import StreamRuntime

    main.init_db()
    runtime = StreamRuntime()
    with SessionLocal() as db:
        active = VideoStream(name="active", stream_url="rtsp://x", status="running")
        stopped = VideoStream(name="stopped", stream_url="rtsp://x", status="stopped")
        db.add_all([active, stopped])
        db.commit()
        active_id, stopped_id = active.id, stopped.id
    stop = threading.Event()
    thread = threading.Thread(target=lambda: (stop.wait(), runtime._mark_stopped(active_id)))
    runtime._threads[active_id] = thread
    runtime._stop_events[active_id] = stop
    thread.start()
    runtime.stop_all()
    assert not thread.is_alive()
    with SessionLocal() as db:
        assert db.get(VideoStream, active_id).status == "starting"
        assert db.get(VideoStream, stopped_id).status == "stopped"


def test_migration_adds_real_read_heartbeat_to_legacy_streams(monkeypatch, tmp_path):
    from sqlalchemy import inspect, text

    main = load_app(monkeypatch, tmp_path, "legacy-stream-heartbeat")
    from app.db.session import engine

    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE video_streams (id CHAR(36) PRIMARY KEY, name VARCHAR)"))
    main.init_db()
    columns = {column["name"] for column in inspect(engine).get_columns("video_streams")}
    assert {"last_frame_read_at", "consecutive_read_failures"} <= columns


def test_warmup_read_updates_heartbeat_without_saving_any_image(monkeypatch, tmp_path):
    import threading

    main = load_app(monkeypatch, tmp_path, "empty-heartbeat", STREAM_WARMUP_FRAMES="3")
    from app.db.session import SessionLocal
    from app.models.media import Image, VideoStream
    from app.services.stream_runtime import StreamRuntime

    main.init_db()
    stop = threading.Event()

    class Capture:
        def isOpened(self):
            return True

        def read(self):
            stop.set()
            return True, object()

        def release(self):
            pass

    monkeypatch.setattr("app.services.stream_runtime.CaptureProcess", lambda *args: Capture())
    with SessionLocal() as db:
        stream = VideoStream(name="warmup", stream_url="rtsp://x")
        db.add(stream)
        db.commit()
        stream_id = stream.id
    StreamRuntime()._run_capture_loop(stream_id, stop)
    with SessionLocal() as db:
        assert db.get(VideoStream, stream_id).last_frame_read_at is not None
        assert db.query(Image).count() == 0


def test_unreapable_decoder_does_not_crash_the_capture_thread(monkeypatch, tmp_path):
    """capture.release() can raise if the decoder process survives even SIGKILL.

    An uncaught exception there would silently kill the whole capture thread without ever
    marking the stream stopped, breaking the automatic reconnect loop until someone notices
    and restarts it by hand.
    """

    import threading

    main = load_app(monkeypatch, tmp_path, "unkillable-decoder")
    from app.db.session import SessionLocal
    from app.models.media import VideoStream
    from app.services.stream_runtime import StreamRuntime

    main.init_db()
    stop = threading.Event()

    class Capture:
        def isOpened(self):
            return True

        def read(self):
            stop.set()
            return True, object()

        def release(self):
            raise RuntimeError("Camera decoder could not be reaped; refusing to reconnect")

    monkeypatch.setattr("app.services.stream_runtime.CaptureProcess", lambda *args: Capture())
    with SessionLocal() as db:
        stream = VideoStream(name="unkillable", stream_url="rtsp://x")
        db.add(stream)
        db.commit()
        stream_id = stream.id

    StreamRuntime()._run_capture_loop(stream_id, stop)

    with SessionLocal() as db:
        assert db.get(VideoStream, stream_id).status == "stopped"
