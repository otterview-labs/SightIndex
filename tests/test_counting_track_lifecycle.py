from types import SimpleNamespace

import pytest

from app.services import video_processing
from app.services.frame_processing import Detection
from app.services.video_processing import CountingLine, PersonTrack, VideoProcessingService

FRAME = SimpleNamespace(shape=(100, 100, 3))
LINE = CountingLine(x1=0.5, y1=0.0, x2=0.5, y2=1.0)


def _service(*, idle_seconds: float = 6.0) -> VideoProcessingService:
    service = VideoProcessingService.__new__(VideoProcessingService)
    service.settings = SimpleNamespace(
        line_crossing_track_idle_seconds=idle_seconds,
        line_crossing_track_max_missed_frames=2,
        line_crossing_match_distance=0.5,
        line_crossing_point="center",
    )
    return service


def _detection(center_x: float) -> Detection:
    return Detection(
        bbox={
            "x": center_x * 100 - 5,
            "y": 45,
            "width": 10,
            "height": 10,
        },
        confidence=1.0,
    )


def _step(
    service: VideoProcessingService,
    tracks: dict[int, PersonTrack],
    next_track_id: int,
    *,
    center_x: float | None,
    now: float | None,
    max_idle_seconds: float | None = None,
):
    detections = [] if center_x is None else [_detection(center_x)]
    return service._line_crossings(
        detections=detections,
        frame=FRAME,
        line=LINE,
        tracks=tracks,
        next_track_id=next_track_id,
        now=now,
        max_idle_seconds=max_idle_seconds,
    )


def test_same_visit_counts_once_but_a_new_visit_after_idle_can_count_again():
    service = _service(idle_seconds=6.0)
    tracks: dict[int, PersonTrack] = {}

    crossings, next_track_id = _step(service, tracks, 1, center_x=0.4, now=0.0)
    assert crossings == []
    crossings, next_track_id = _step(service, tracks, next_track_id, center_x=0.6, now=1.0)
    assert len(crossings) == 1

    # Moving the same live track back and forth must not count the same visit again.
    crossings, next_track_id = _step(service, tracks, next_track_id, center_x=0.4, now=2.0)
    assert crossings == []
    crossings, next_track_id = _step(service, tracks, next_track_id, center_x=0.6, now=3.0)
    assert crossings == []

    # An empty frame advances lifecycle time and retires the counted visit.
    crossings, next_track_id = _step(service, tracks, next_track_id, center_x=None, now=9.1)
    assert crossings == []
    assert tracks == {}

    crossings, next_track_id = _step(service, tracks, next_track_id, center_x=0.4, now=9.2)
    assert crossings == []
    crossings, next_track_id = _step(service, tracks, next_track_id, center_x=0.6, now=10.0)
    assert len(crossings) == 1
    assert next_track_id == 3


def test_empty_detections_expire_an_unmatched_track():
    service = _service(idle_seconds=2.0)
    tracks = {
        7: PersonTrack(
            id=7,
            center=(0.4, 0.5),
            side=0.1,
            last_seen_at=10.0,
        )
    }

    crossings, next_track_id = _step(service, tracks, 8, center_x=None, now=12.1)

    assert crossings == []
    assert next_track_id == 8
    assert tracks == {}


def test_slow_sampling_keeps_a_track_for_two_intervals_then_expires_it():
    service = _service(idle_seconds=6.0)
    tracks: dict[int, PersonTrack] = {}

    crossings, next_track_id = _step(
        service,
        tracks,
        1,
        center_x=0.4,
        now=0.0,
        max_idle_seconds=20.0,
    )
    assert crossings == []
    crossings, next_track_id = _step(
        service,
        tracks,
        next_track_id,
        center_x=0.6,
        now=20.0,
        max_idle_seconds=20.0,
    )
    assert len(crossings) == 1

    # A gap longer than the effective two-frame window starts a new visit instead of matching
    # against the old counted track.
    crossings, next_track_id = _step(
        service,
        tracks,
        next_track_id,
        center_x=None,
        now=40.1,
        max_idle_seconds=20.0,
    )
    assert crossings == []
    assert tracks == {}


def test_legacy_track_construction_uses_monotonic_time_when_now_is_omitted(monkeypatch):
    service = _service()
    tracks = {
        1: PersonTrack(
            id=1,
            center=(0.4, 0.5),
            side=0.1,
        )
    }
    monkeypatch.setattr(video_processing.time, "monotonic", lambda: 42.5)

    crossings, next_track_id = _step(service, tracks, 2, center_x=0.6, now=None)

    assert len(crossings) == 1
    assert next_track_id == 2
    assert tracks[1].last_seen_at == 42.5


def test_clock_rollback_rebases_valid_tracks_and_drops_non_finite_state():
    service = _service(idle_seconds=6.0)
    tracks = {
        1: PersonTrack(
            id=1,
            center=(0.4, 0.5),
            side=0.1,
            counted=True,
            last_seen_at=100.0,
        ),
        2: PersonTrack(
            id=2,
            center=(float("nan"), 0.5),
            side=0.1,
            last_seen_at=100.0,
        ),
        3: PersonTrack(
            id=3,
            center=(0.3, 0.5),
            side=float("inf"),
            last_seen_at=100.0,
        ),
        4: PersonTrack(
            id=4,
            center=(0.2, 0.5),
            side=0.2,
            last_seen_at=float("nan"),
        ),
    }

    crossings, next_track_id = _step(service, tracks, 5, center_x=None, now=90.0)

    assert crossings == []
    assert next_track_id == 5
    assert set(tracks) == {1}
    assert tracks[1].last_seen_at == 90.0

    # The rebased counted track remains the same visit and then expires normally.
    crossings, next_track_id = _step(service, tracks, next_track_id, center_x=0.6, now=90.1)
    assert crossings == []
    _step(service, tracks, next_track_id, center_x=None, now=96.2)
    assert tracks == {}


@pytest.mark.parametrize("invalid_now", [float("nan"), float("inf")])
def test_invalid_explicit_time_falls_back_to_monotonic(monkeypatch, invalid_now):
    service = _service()
    monkeypatch.setattr(video_processing.time, "monotonic", lambda: 123.0)
    tracks: dict[int, PersonTrack] = {}

    crossings, next_track_id = _step(service, tracks, 1, center_x=0.4, now=invalid_now)

    assert crossings == []
    assert next_track_id == 2
    assert tracks[1].last_seen_at == 123.0


def test_explicit_sampling_window_does_not_disable_missed_frame_expiry():
    service = _service()
    tracks: dict[int, PersonTrack] = {}
    _, next_track_id = _step(service, tracks, 1, center_x=0.4, now=0.0, max_idle_seconds=20.0)

    _step(service, tracks, next_track_id, center_x=None, now=1.0, max_idle_seconds=20.0)
    assert tracks[1].missed_frames == 1

    _step(service, tracks, next_track_id, center_x=None, now=2.0, max_idle_seconds=20.0)
    assert tracks == {}


def test_public_idle_setting_takes_precedence_over_legacy_setting():
    service = _service(idle_seconds=2.0)
    service.settings.counting_track_idle_seconds = 60.0
    tracks = {1: PersonTrack(id=1, center=(0.4, 0.5), side=0.1, last_seen_at=0.0)}

    crossings, next_track_id = _step(service, tracks, 2, center_x=0.6, now=2.0)

    assert crossings == []
    assert next_track_id == 3
    assert set(tracks) == {2}


@pytest.mark.parametrize("invalid_coordinate", [float("nan"), float("inf")])
def test_non_finite_detection_cannot_match_or_create_a_track(invalid_coordinate):
    service = _service()
    tracks = {1: PersonTrack(id=1, center=(0.4, 0.5), side=0.1, last_seen_at=0.0)}

    crossings, next_track_id = _step(service, tracks, 2, center_x=invalid_coordinate, now=1.0)

    assert crossings == []
    assert next_track_id == 2
    assert tracks[1].center == (0.4, 0.5)
    assert tracks[1].last_seen_at == 0.0
    assert tracks[1].missed_frames == 1
