"""Exercise the stream integration, not only the standalone retention tracker."""

import pytest

from app.services.appearance_tracker import AppearanceTracker
from app.services.frame_processing import Detection
from app.services.stream_runtime import StreamRuntime

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def test_later_clearer_person_frames_survive_dedupe_and_keep_metadata(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr("app.services.stream_runtime.time.monotonic", lambda: clock[0])
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
        min_sample_interval_seconds=2.0,
        quality_improvement_ratio=0.15,
    )
    stripes = np.where((np.indices((256, 128))[1] // 4) % 2, 192, 64).astype(np.uint8)
    person = np.repeat(stripes[:, :, None], 3, axis=2)
    detection = Detection(bbox={"x": 20, "y": 20, "width": 128, "height": 256}, confidence=0.9)
    originals = dict(detection.bbox)
    qualities = []
    for now, kernel, sigma in [(0.0, 11, 4.0), (2.0, 5, 1.5), (4.0, 1, 0.0)]:
        clock[0] = now
        frame = np.full((300, 200, 3), 128, dtype=np.uint8)
        frame[20:276, 20:148] = cv2.GaussianBlur(person, (kernel, kernel), sigma)
        retained = StreamRuntime._first_sightings([detection], frame, tracker)
        assert len(retained) == 1
        qualities.append(retained[0].bbox["roi_quality"]["score"])
        assert retained[0].bbox["roi_quality"]["width"] == 128
    assert qualities[0] < qualities[1] < qualities[2]
    assert detection.bbox == originals, "input detections must not be mutated"
    clock[0] = 6.0
    assert StreamRuntime._first_sightings([detection], frame, tracker) == []


def test_missing_pixel_measurement_preserves_legacy_retention(monkeypatch) -> None:
    class Frame:
        shape = (300, 200, 3)

    monkeypatch.setattr("app.services.stream_runtime.time.monotonic", lambda: 1.0)
    tracker = AppearanceTracker(match_distance=0.32, idle_seconds=6.0, max_seconds=300.0)
    detection = Detection(bbox={"x": 20, "y": 20, "width": 50, "height": 100}, confidence=0.9)
    retained = StreamRuntime._first_sightings([detection], Frame(), tracker)
    assert len(retained) == 1
    assert isinstance(retained[0].bbox["sampling_visit_token"], str)
    assert "roi_quality" not in retained[0].bbox
    assert retained[0].confidence == detection.confidence
    assert StreamRuntime._first_sightings([detection], Frame(), tracker) == []
