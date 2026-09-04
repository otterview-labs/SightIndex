"""Relative stature: what the floor line supports saying, and what it refuses.

The fit itself is exercised through a fake camera whose crops are generated from a known line,
so a person planted above that line has to come back tall.
"""
import uuid

import numpy as np
import pytest

from app.services.stature import StatureService, reset_cache

FRAME_W, FRAME_H = 2560, 1920
INTERCEPT, SLOPE = 60.0, 0.42


@pytest.fixture(autouse=True)
def clear_cache():
    reset_cache()
    yield
    reset_cache()


def box(foot_y, height, *, confidence=0.9, x=800.0, width=300.0):
    return {
        "x": x,
        "y": foot_y - height,
        "width": width,
        "height": height,
        "confidence": confidence,
        "frame_width": FRAME_W,
        "frame_height": FRAME_H,
    }


def crowd(n=400, spread=0.0, seed=1):
    """Boxes drawn from the true line, optionally with per-person height variation.

    Feet land across the width as well as the depth, or the surface's column terms would be
    fitted from a single column and the recovery test would prove nothing about them.
    """

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        foot_y = float(rng.uniform(400, 1800))
        foot_x = float(rng.uniform(500, 2000))
        scale = 1.0 + (rng.normal(0, spread) if spread else 0.0)
        height = (INTERCEPT + SLOPE * foot_y) * scale
        rows.append((box(foot_y, height, x=foot_x - 150.0), "/data/frames/f.jpg"))
    return rows


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class FakeDB:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, _statement):
        return FakeResult(self.rows)


class FakeSettings:
    data_dir = "/nonexistent"


def service_for(rows):
    return StatureService(FakeDB(rows), FakeSettings())


# --- the fit ---------------------------------------------------------------------------


def test_the_floor_surface_predicts_what_the_crowd_was_drawn_from():
    """Checked through predictions rather than coefficients: the surface has six of them and
    only their combined effect is meaningful."""

    calibration = service_for(crowd()).calibration(uuid.uuid4())

    assert calibration is not None
    for foot_y in (500.0, 1100.0, 1700.0):
        assert calibration.predict(1200.0, foot_y) == pytest.approx(
            INTERCEPT + SLOPE * foot_y, rel=0.02
        )


def test_a_camera_with_too_few_crops_is_not_calibrated():
    """A line through a handful of points would give every one of them a confident ratio."""

    assert service_for(crowd(n=30)).calibration(uuid.uuid4()) is None


def test_the_ratio_does_not_track_where_in_the_room_somebody_stands():
    """The failure this surface exists to prevent: a bare line left the ratio correlated with
    the column at 0.714, and that correlation looked like identity."""

    rows = crowd(n=600, spread=0.06, seed=5)
    service = service_for(rows)
    camera = uuid.uuid4()
    service.calibration(camera)

    positions, ratios = [], []
    for bbox, _url in rows:
        described = service.describe(bbox, camera)
        if described:
            positions.append(bbox["x"] + bbox["width"] / 2.0)
            ratios.append(described["ratio"])

    assert abs(np.corrcoef(positions, ratios)[0, 1]) < 0.2


def test_frame_size_is_taken_from_the_capture_rather_than_reopening_a_frame():
    calibration = service_for(crowd()).calibration(uuid.uuid4())

    assert (calibration.frame_width, calibration.frame_height) == (FRAME_W, FRAME_H)


def test_a_camera_that_was_never_named_has_no_line():
    assert service_for(crowd()).calibration(None) is None


# --- reading one person ----------------------------------------------------------------


def calibrated(spread=0.06):
    service = service_for(crowd(spread=spread))
    camera = uuid.uuid4()
    service.calibration(camera)
    return service, camera


def test_someone_taller_than_the_line_predicts_reads_tall():
    service, camera = calibrated()
    foot_y = 1000.0
    expected = INTERCEPT + SLOPE * foot_y

    assert service.describe(box(foot_y, expected * 1.15), camera)["band"] == "tall"


def test_someone_shorter_than_the_line_predicts_reads_short():
    service, camera = calibrated()
    foot_y = 1000.0
    expected = INTERCEPT + SLOPE * foot_y

    assert service.describe(box(foot_y, expected * 0.85), camera)["band"] == "short"


def test_the_same_person_reads_the_same_near_and_far():
    """The whole point: the ratio must not track where in the room somebody is standing."""

    service, camera = calibrated()
    near, far = 1700.0, 500.0
    tall = 1.12

    near_ratio = service.describe(box(near, (INTERCEPT + SLOPE * near) * tall), camera)["ratio"]
    far_ratio = service.describe(box(far, (INTERCEPT + SLOPE * far) * tall), camera)["ratio"]

    assert near_ratio == pytest.approx(far_ratio, abs=0.02)


def test_a_box_clipped_by_the_frame_edge_is_refused():
    """It stops where the frame stops, so its height measures the frame, not the person."""

    service, camera = calibrated()
    tall_enough = INTERCEPT + SLOPE * FRAME_H

    assert service.describe(box(FRAME_H - 1, tall_enough), camera) is None
    assert service.describe(box(900.0, 890.0, x=1.0), camera) is None


def test_a_low_confidence_detection_is_refused():
    service, camera = calibrated()

    assert service.describe(box(1000.0, 480.0, confidence=0.4), camera) is None


def test_an_impossible_ratio_is_a_bad_box_not_a_giant():
    service, camera = calibrated()

    assert service.describe(box(1000.0, 30.0), camera) is None


def test_an_uncalibrated_camera_yields_nothing_rather_than_a_guess():
    service = service_for(crowd(n=30))

    assert service.describe(box(1000.0, 480.0), uuid.uuid4()) is None


def test_bands_are_relative_to_this_doorway_crowd():
    """"Tall" has to mean tall for the people who walk through here."""

    service, camera = calibrated()
    calibration = service.calibration(camera)

    assert calibration.rank(1.0) == pytest.approx(50, abs=8)
    # Below everyone this camera has seen, and above everyone, clamp rather than run off the end.
    assert calibration.rank(calibration.quantiles[0] - 0.05) == 0
    assert calibration.rank(calibration.quantiles[-1] + 0.05) == 100


def test_a_rank_is_published_so_two_cameras_can_be_compared():
    """The raw ratio spans 0.935-1.062 at one door here and 0.882-1.150 at the other, so only
    the rank means the same thing on both."""

    service, camera = calibrated()

    tall = service.describe(box(1000.0, (INTERCEPT + SLOPE * 1000.0) * 1.15), camera)

    assert tall["percentile"] >= 80
    assert tall["band"] == "tall"


# --- how it reaches the panel and the query ---------------------------------------------


def test_the_band_reaches_the_attribute_panel():
    from app.services.observation_index import _chinese_attribute_labels

    labels = _chinese_attribute_labels(
        {"object_type": "person", "stature": {"ratio": 1.12, "band": "tall"}}
    )

    assert labels["身高"] == "偏高"


def test_an_unmeasured_stature_leaves_no_row():
    from app.services.observation_index import _chinese_attribute_labels

    assert "身高" not in _chinese_attribute_labels({"object_type": "person"})


@pytest.mark.parametrize(
    ("query", "expected"),
    [("高个子男的", "tall"), ("矮个子", "short"), ("穿黑衣服的", None)],
)
def test_a_height_word_in_the_query_becomes_a_condition(query, expected):
    from app.services.search import StructuredSearchService

    assert StructuredSearchService._query_stature(query) == expected


def test_the_stature_condition_reads_the_nested_band():
    from app.services.search import StructuredSearchService

    service = StructuredSearchService.__new__(StructuredSearchService)

    assert service._attribute_values({"stature": {"band": "tall"}}, {}, "stature") == ["tall"]
