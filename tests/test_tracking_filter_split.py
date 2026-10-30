"""Tracking must not inherit the crop-storage filter.

The crop-quality rules (min confidence, min box size, whole body) exist to keep bad crops out
of the library. Applied to counting they also delete the detections a walk-in depends on while
the frame edge clips it, so the track dies at the one moment it has to stay continuous. These
tests pin the split down: track wide, store strict.
"""

import pytest

FRAME_WIDTH, FRAME_HEIGHT = 1280, 720


def _service(tmp_path, **overrides):
    from app.config.settings import Settings
    from app.services.frame_processing import FrameProcessingService, WholeFramePersonDetector

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        # "whole_frame" is what the rest of the suite uses: the default "yolo" builds an
        # Ultralytics detector at construction time, which the AGX venv does not have.
        person_detector="whole_frame",
        **overrides,
    )
    return FrameProcessingService(db=None, settings=settings, detector=WholeFramePersonDetector())


def _detection(x, y, width, height, confidence=0.9):
    from app.services.frame_processing import Detection

    return Detection(
        bbox={"x": x, "y": y, "width": width, "height": height}, confidence=confidence
    )


def test_a_person_clipped_by_the_frame_edge_is_tracked_but_not_stored(tmp_path):
    service = _service(tmp_path)
    clipped = _detection(0, 100, 200, 400)

    assert service.quality_filter_detections([clipped], FRAME_WIDTH, FRAME_HEIGHT) == []
    assert service.tracking_filter_detections([clipped], FRAME_WIDTH, FRAME_HEIGHT) == [clipped]


def test_tracking_accepts_boxes_below_the_crop_confidence_floor(tmp_path):
    """A half-hidden person is still a person. The crop rule is the one that can wait."""

    service = _service(tmp_path, person_crop_min_confidence=0.70)
    weak = _detection(300, 100, 200, 400, confidence=0.4)

    assert service.quality_filter_detections([weak], FRAME_WIDTH, FRAME_HEIGHT) == []
    assert service.tracking_filter_detections([weak], FRAME_WIDTH, FRAME_HEIGHT) == [weak]


def test_tracking_accepts_everything_the_detector_reports_by_default(tmp_path):
    """The detector owns the confidence floor. A second one here only loses people.

    This is the default that the AGX measurement decided. On 2026-10-16 the deployed filter
    dropped 592 detections in 24 hours -- 8.5% of all detections, and the source of 203 frames
    that had people in them yet reached the counter empty -- and every one of them sat between
    the detector floor and the crop floor. With tracking_min_confidence at 0.10 the split
    recovered none of them, which is the whole reason the default is 0.0.
    """

    from app.config.settings import Settings

    # Read the declared default rather than an instantiated one: some tests in this suite leave
    # .env values in os.environ, and pydantic-settings reads os.environ even under _env_file=None.
    assert Settings.model_fields["tracking_min_confidence"].default == 0.0

    # The worst case measured on the production camera: a person the size of a full doorway
    # crossing, low-confidence enough to fail the crop rule.
    service = _service(tmp_path, person_crop_min_confidence=0.70, tracking_min_confidence=0.0)
    weak = _detection(300, 100, 153, 370, confidence=0.072)

    assert service.quality_filter_detections([weak], FRAME_WIDTH, FRAME_HEIGHT) == []
    assert service.tracking_filter_detections([weak], FRAME_WIDTH, FRAME_HEIGHT) == [weak]


def test_tracking_still_has_its_own_confidence_floor(tmp_path):
    """Looser than the crop rule, not absent: a deployment may still raise it deliberately."""

    service = _service(tmp_path, tracking_min_confidence=0.3)
    noise = _detection(300, 100, 200, 400, confidence=0.2)

    assert service.tracking_filter_detections([noise], FRAME_WIDTH, FRAME_HEIGHT) == []


def test_tracking_can_require_a_whole_body_when_a_deployment_asks(tmp_path):
    service = _service(tmp_path, tracking_require_whole_body=True)
    clipped = _detection(0, 100, 200, 400)

    assert service.tracking_filter_detections([clipped], FRAME_WIDTH, FRAME_HEIGHT) == []


def test_storable_crossings_keeps_the_indices_of_the_crops_it_accepts(tmp_path):
    service = _service(tmp_path)
    detections = [
        _detection(0, 100, 200, 400),    # clipped: tracked, not stored
        _detection(300, 100, 200, 400),  # clean: stored
        _detection(0, 100, 220, 400),    # clipped
    ]

    accepted, indices = service.storable_crossings(detections, [0, 1, 2], FRAME_WIDTH, FRAME_HEIGHT)

    assert accepted == [detections[1]]
    assert indices == [1]


def test_a_crossing_is_still_countable_with_nothing_to_store(tmp_path):
    """The count is what the operator asked for; the crop is a convenience."""

    service = _service(tmp_path)
    clipped = _detection(0, 100, 200, 400)

    accepted, indices = service.storable_crossings([clipped], [0], FRAME_WIDTH, FRAME_HEIGHT)

    assert accepted == []
    assert indices == []


def test_a_crossing_index_outside_the_detection_list_is_ignored(tmp_path):
    service = _service(tmp_path)
    clean = _detection(300, 100, 200, 400)

    accepted, indices = service.storable_crossings([clean], [0, 7], FRAME_WIDTH, FRAME_HEIGHT)

    assert accepted == [clean]
    assert indices == [0]


def test_a_person_clipped_by_the_edge_still_crosses_the_line(tmp_path):
    """The tracker reads the pre-filter boxes, so a clipped walker still produces a crossing."""

    numpy = pytest.importorskip("numpy")
    from app.services.video_processing import CountingLine, VideoProcessingService

    service = _service(tmp_path)
    counter = VideoProcessingService(db=None, settings=service.settings)

    frame = numpy.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=numpy.uint8)
    line = CountingLine(x1=0.0, y1=0.5, x2=1.0, y2=0.5)
    tracks: dict[int, object] = {}

    # The foot-point sits at 45% then 55% of the frame height: one small step over a mid-frame
    # line. Both boxes stay on the left edge, which is what the crop rule rejects.
    above = _detection(0, 100, 200, 224)
    below = _detection(0, 100, 200, 296)

    assert service.quality_filter_detections([above], FRAME_WIDTH, FRAME_HEIGHT) == []
    assert service.quality_filter_detections([below], FRAME_WIDTH, FRAME_HEIGHT) == []

    crossings, next_track_id = counter._line_crossings(
        detections=service.tracking_filter_detections([above], FRAME_WIDTH, FRAME_HEIGHT),
        frame=frame,
        line=line,
        tracks=tracks,
        next_track_id=1,
        observed_at=0.0,
    )
    assert crossings == []

    tracking_below = service.tracking_filter_detections([below], FRAME_WIDTH, FRAME_HEIGHT)
    crossings, _ = counter._line_crossings(
        detections=tracking_below,
        frame=frame,
        line=line,
        tracks=tracks,
        next_track_id=next_track_id,
        observed_at=1.0,
    )

    assert [crossing.direction for crossing in crossings] == ["a_to_b"]
    accepted, _ = service.storable_crossings(
        tracking_below,
        [crossing.detection_index for crossing in crossings],
        FRAME_WIDTH,
        FRAME_HEIGHT,
    )
    assert accepted == []


def test_process_video_path_counts_an_edge_clipped_person_without_storing_a_crop(
    monkeypatch, tmp_path
):
    """End to end: the crossing lands as a counting event, and the crop list stays empty.

    Before the split this produced no event at all, because the clipped boxes were removed
    before the tracker ran.
    """

    from datetime import datetime

    try:
        from datetime import UTC
    except ImportError:  # The AGX deployment still runs 3.10; the app modules shim the same way.
        from datetime import timezone

        UTC = timezone.utc

    from fastapi.testclient import TestClient
    from test_reid import load_app

    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    main = load_app(
        monkeypatch,
        tmp_path,
        "video-track-clipped",
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

    video_path = tmp_path / "clipped.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 1.0, (100, 100)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG encoder is unavailable")
    try:
        for _ in range(2):
            writer.write(numpy.zeros((100, 100, 3), dtype=numpy.uint8))
    finally:
        writer.release()

    heights = iter([40, 60])

    def detect(image_path):
        # x=0 holds the box against the left edge for the whole crossing.
        return [
            Detection(
                bbox={"x": 0, "y": 0, "width": 20, "height": next(heights)},
                confidence=0.95,
            )
        ]

    with TestClient(main.create_app()), SessionLocal() as db:
        service = VideoProcessingService(db, main.get_settings())
        monkeypatch.setattr(service.processor, "detect_image_path", detect)
        result = service.process_video_path(
            video_path,
            "/data/videos/clipped.avi",
            max_frames=2,
            counting_line=CountingLine(0.0, 0.5, 1.0, 0.5),
            captured_at=datetime(2026, 10, 15, tzinfo=UTC),
        )

    assert result.frames_sampled == 2
    assert result.counting_events_created == 1
    assert result.crops_created == 0
    # The evidence frame is still kept, even though nothing could be cropped out of it.
    assert result.images_created == 1
