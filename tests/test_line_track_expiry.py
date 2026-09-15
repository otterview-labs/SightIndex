from types import SimpleNamespace

import pytest


@pytest.fixture
def counter():
    from app.config.settings import Settings
    from app.services.video_processing import VideoProcessingService

    service = VideoProcessingService.__new__(VideoProcessingService)
    service.settings = Settings(
        _env_file=None,
        line_crossing_track_idle_seconds=6.0,
        line_crossing_track_max_missed_frames=2,
    )
    return service


def _observe(counter, tracks, next_id, horizontal, timestamp):
    from app.services.frame_processing import Detection
    from app.services.video_processing import CountingLine

    detections = (
        []
        if horizontal is None
        else [
            Detection(
                bbox={"x": horizontal, "y": 40, "width": 10, "height": 10},
                confidence=1.0,
            )
        ]
    )
    return counter._line_crossings(
        detections,
        SimpleNamespace(shape=(100, 100, 3)),
        CountingLine(0.5, 0.0, 0.5, 1.0),
        tracks,
        next_id,
        observed_at=timestamp,
    )


def test_separate_visitors_each_count_after_empty_frames(counter):
    tracks = {}
    next_id = 1
    count = 0
    for timestamp, horizontal in enumerate([40, 50, None, None, 40, 50]):
        crossings, next_id = _observe(counter, tracks, next_id, horizontal, timestamp)
        count += len(crossings)
    assert count == 2
    assert next_id == 3


def test_single_missed_frame_keeps_a_live_track(counter):
    tracks = {}
    _, next_id = _observe(counter, tracks, 1, 40, 0.0)
    _, next_id = _observe(counter, tracks, next_id, None, 1.0)
    crossings, next_id = _observe(counter, tracks, next_id, 50, 2.0)
    assert len(crossings) == 1
    assert next_id == 2
    assert tracks[1].missed_frames == 0


def test_idle_gap_cannot_create_a_crossing_from_an_old_position(counter):
    tracks = {}
    _, next_id = _observe(counter, tracks, 1, 40, 0.0)
    crossings, next_id = _observe(counter, tracks, next_id, 50, 6.0)
    assert crossings == []
    assert next_id == 3
    assert 1 not in tracks


def test_continuous_oscillation_does_not_recount_one_visit(counter):
    tracks = {}
    next_id = 1
    count = 0
    for timestamp, horizontal in enumerate([40, 50, 40, 50]):
        crossings, next_id = _observe(counter, tracks, next_id, horizontal, timestamp)
        count += len(crossings)
    assert count == 1


def test_video_processing_counts_two_visitors_separated_by_empty_frames(monkeypatch, tmp_path):
    from datetime import UTC, datetime

    from fastapi.testclient import TestClient
    from test_reid import load_app

    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    main = load_app(
        monkeypatch,
        tmp_path,
        "video-track-expiry",
        MILVUS_ENABLED="false",
        REID_ENABLED="false",
        VECTOR_INDEX_ON_INGEST="false",
        APPEARANCE_TONE_ON_INGEST="false",
        FACE_RECOGNITION_ON_INGEST="false",
        VLM_STRUCTURED_ON_INGEST="false",
    )
    from app.db.session import SessionLocal
    from app.services.frame_processing import Detection
    from app.services.video_processing import CountingLine, VideoProcessingService

    video_path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 1.0, (100, 100)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG encoder is unavailable")
    try:
        for _ in range(6):
            writer.write(numpy.zeros((100, 100, 3), dtype=numpy.uint8))
    finally:
        writer.release()
    positions = iter([40, 50, None, None, 40, 50])

    def detect(image_path):
        horizontal = next(positions)
        return (
            []
            if horizontal is None
            else [
                Detection(
                    bbox={"x": horizontal, "y": 40, "width": 10, "height": 10},
                    confidence=1.0,
                )
            ]
        )

    with TestClient(main.create_app()), SessionLocal() as db:
        service = VideoProcessingService(db, main.get_settings())
        monkeypatch.setattr(service.processor, "detect_image_path", detect)
        result = service.process_video_path(
            video_path,
            "/data/videos/synthetic.avi",
            max_frames=6,
            counting_line=CountingLine(0.5, 0.0, 0.5, 1.0),
            captured_at=datetime(2026, 9, 7, tzinfo=UTC),
        )

    assert result.frames_sampled == 6
    assert result.counting_events_created == 2
    assert result.images_created == 2
    assert result.crops_created == 2
