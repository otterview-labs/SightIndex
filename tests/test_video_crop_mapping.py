import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.models.media import PersonCrop
from app.services.frame_processing import Detection
from app.services.video_processing import LineCrossing, VideoProcessingService


class _Db:
    def __init__(self) -> None:
        self.added = []

    def add(self, item) -> None:
        self.added.append(item)


def _crop(*, source_index=None):
    bbox = {}
    if source_index is not None:
        bbox["source_detection_index"] = source_index
    return SimpleNamespace(
        id=uuid.uuid4(),
        bbox=bbox,
        person_id=None,
        identity_is_protected=False,
        camera_id=None,
        location_id=None,
    )


def test_partial_crop_failure_keeps_the_successful_crop_on_its_original_crossing():
    service = VideoProcessingService.__new__(VideoProcessingService)
    service.db = _Db()
    service._recognition_event_for_crop = lambda _crop: None
    detections = [
        Detection(bbox={"x": 10, "y": 10, "width": 20, "height": 40}, confidence=1),
        Detection(bbox={"x": 60, "y": 10, "width": 20, "height": 40}, confidence=1),
    ]
    crossings = [
        LineCrossing(detection_index=0, direction="a_to_b"),
        LineCrossing(detection_index=1, direction="a_to_b"),
    ]

    indexed_detections = service._crossing_detections(detections, crossings)
    assert [detection.source_index for detection in indexed_detections] == [0, 1]

    # Simulate crop 0 failing while crop 1 succeeds and preserves its source index.
    second_crop = _crop(source_index=indexed_detections[1].source_index)
    crops_by_detection_index = service._crops_by_detection_index([second_crop])
    created = service._create_line_crossing_events(
        crossings=crossings,
        crops_by_detection_index=crops_by_detection_index,
        counted_at=datetime(2026, 10, 6, tzinfo=UTC),
        camera_id=None,
        location_id=None,
    )

    assert created == 2
    assert service.db.added[0].crop_id is None
    assert service.db.added[1].crop_id == second_crop.id


def test_partial_legacy_crop_list_without_source_indices_abstains_from_mapping():
    service = VideoProcessingService.__new__(VideoProcessingService)
    service.db = _Db()
    service._recognition_event_for_crop = lambda _crop: None
    service._line_crossings = lambda **_kwargs: (
        [LineCrossing(detection_index=1, direction="a_to_b")],
        3,
    )
    detections = [
        Detection(bbox={"x": 10, "y": 10, "width": 20, "height": 40}, confidence=1),
        Detection(bbox={"x": 60, "y": 10, "width": 20, "height": 40}, confidence=1),
    ]
    only_first_crop = _crop()

    count, next_track_id = service._count_line_crossings(
        detections=detections,
        frame=SimpleNamespace(shape=(100, 100, 3)),
        line=SimpleNamespace(),
        tracks={},
        next_track_id=1,
        counted_at=datetime(2026, 10, 6, tzinfo=UTC),
        camera_id=None,
        location_id=None,
        crops=[only_first_crop],
    )

    assert count == 1
    assert next_track_id == 3
    assert service.db.added[0].crop_id is None


def test_complete_legacy_crop_list_keeps_positional_compatibility():
    service = VideoProcessingService.__new__(VideoProcessingService)
    service.db = _Db()
    service._recognition_event_for_crop = lambda _crop: None
    service._line_crossings = lambda **_kwargs: (
        [LineCrossing(detection_index=1, direction="a_to_b")],
        3,
    )
    detections = [
        Detection(bbox={"x": 10, "y": 10, "width": 20, "height": 40}, confidence=1),
        Detection(bbox={"x": 60, "y": 10, "width": 20, "height": 40}, confidence=1),
    ]
    crops = [_crop(), _crop()]

    count, _ = service._count_line_crossings(
        detections=detections,
        frame=SimpleNamespace(shape=(100, 100, 3)),
        line=SimpleNamespace(),
        tracks={},
        next_track_id=1,
        counted_at=datetime(2026, 10, 6, tzinfo=UTC),
        camera_id=None,
        location_id=None,
        crops=crops,
    )

    assert count == 1
    assert service.db.added[0].crop_id == crops[1].id


@pytest.mark.parametrize("person_id_source", ["manual", None])
def test_protected_identity_discards_a_conflicting_recognition(person_id_source):
    service = VideoProcessingService.__new__(VideoProcessingService)
    service.db = _Db()
    selected_person = uuid.uuid4()
    crop = PersonCrop(
        id=uuid.uuid4(),
        image_id=uuid.uuid4(),
        crop_url="/data/crops/protected.jpg",
        bbox={"source_detection_index": 1},
        person_id=selected_person,
        person_id_source=person_id_source,
    )
    recognition = SimpleNamespace(
        id=uuid.uuid4(),
        person_id=uuid.uuid4(),
        unknown_cluster_id=uuid.uuid4(),
    )
    service._recognition_event_for_crop = lambda _crop: recognition

    count = service._create_line_crossing_events(
        crossings=[LineCrossing(detection_index=1, direction="a_to_b")],
        crops_by_detection_index={1: crop},
        counted_at=datetime(2026, 10, 6, tzinfo=UTC),
        camera_id=None,
        location_id=None,
    )

    assert count == 1
    event = service.db.added[0]
    assert event.crop_id == crop.id
    assert event.person_id == selected_person
    assert event.recognition_event_id is None
    assert event.unknown_cluster_id is None


def test_explicit_association_never_falls_back_to_position_for_other_crops():
    service = VideoProcessingService.__new__(VideoProcessingService)
    service.db = _Db()
    service._recognition_event_for_crop = lambda _crop: None
    service._line_crossings = lambda **_kwargs: (
        [LineCrossing(detection_index=1, direction="a_to_b")],
        3,
    )
    detections = [
        Detection(bbox={"x": 10, "y": 10, "width": 20, "height": 40}, confidence=1),
        Detection(bbox={"x": 60, "y": 10, "width": 20, "height": 40}, confidence=1),
    ]

    count, _ = service._count_line_crossings(
        detections=detections,
        frame=SimpleNamespace(shape=(100, 100, 3)),
        line=SimpleNamespace(),
        tracks={},
        next_track_id=1,
        counted_at=datetime(2026, 10, 6, tzinfo=UTC),
        camera_id=None,
        location_id=None,
        crops=[_crop(source_index=0), _crop()],
    )

    assert count == 1
    assert service.db.added[0].crop_id is None
