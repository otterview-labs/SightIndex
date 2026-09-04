"""Letting a genuine crossing through, without letting same-camera noise through with it."""
import uuid

import pytest

from app.api.reid import _admit, _reserve_camera_slots
from app.config.settings import Settings
from app.schemas.reid import ReidMatchItem
from app.services.reid_index import ReidIndexService

DOOR_A, DOOR_B, DOOR_C = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path,
        reid_min_score=0.5,
        reid_min_score_cross_camera=0.42,
        reid_camera_quota=3,
    )


def _item(camera, score):
    return ReidMatchItem(crop_id=uuid.uuid4(), score=score, camera_id=camera)


def test_a_crossing_above_the_bar_survives(settings):
    """The band real crossings actually land in; the single 0.5 bar dropped all of them."""

    admitted = _admit([_item(DOOR_B, 0.46)], settings, query_camera=DOOR_A)

    assert len(admitted) == 1


def test_the_same_camera_is_still_held_to_the_strict_bar(settings):
    """Relaxing everything would fill the answer with weak matches at the door searched from."""

    admitted = _admit([_item(DOOR_A, 0.45), _item(DOOR_A, 0.55)], settings, query_camera=DOOR_A)

    assert [round(item.score, 2) for item in admitted] == [0.55]


def test_an_uploaded_photo_has_no_home_camera(settings):
    """Nothing is same-camera, so every match is a crossing and gets the crossing bar."""

    admitted = _admit([_item(DOOR_A, 0.45), _item(DOOR_B, 0.45)], settings, query_camera=None)

    assert len(admitted) == 2


def test_a_weaker_camera_gets_seats_it_could_never_win_on_score():
    """A crossing at 0.45 ranks below twenty same-camera visits and is never seen without this."""

    crowd = [_item(DOOR_A, 0.90 - index * 0.01) for index in range(20)]
    crossing = [_item(DOOR_B, 0.46), _item(DOOR_B, 0.45)]

    chosen = _reserve_camera_slots([*crowd, *crossing], limit=10, quota=3)

    assert len(chosen) == 10
    assert sum(1 for item in chosen if item.camera_id == DOOR_B) == 2


def test_the_seats_come_from_the_most_crowded_camera():
    crowd_a = [_item(DOOR_A, 0.90 - index * 0.01) for index in range(8)]
    crowd_b = [_item(DOOR_B, 0.80 - index * 0.01) for index in range(2)]
    newcomer = [_item(DOOR_C, 0.44)]

    chosen = _reserve_camera_slots([*crowd_a, *crowd_b, *newcomer], limit=10, quota=3)

    assert sum(1 for item in chosen if item.camera_id == DOOR_C) == 1
    assert sum(1 for item in chosen if item.camera_id == DOOR_B) == 2, "the smaller camera paid"


def test_the_strongest_matches_are_never_displaced():
    crowd = [_item(DOOR_A, 0.99 - index * 0.01) for index in range(20)]
    crossing = [_item(DOOR_B, 0.43)]

    chosen = _reserve_camera_slots([*crowd, *crossing], limit=5, quota=3)

    assert round(max(item.score for item in chosen), 2) == 0.99
    assert chosen == sorted(chosen, key=lambda item: item.score, reverse=True)


def test_one_camera_is_left_exactly_as_ranked():
    items = [_item(DOOR_A, 0.90 - index * 0.01) for index in range(20)]

    chosen = _reserve_camera_slots(items, limit=6, quota=3)

    assert [round(i.score, 2) for i in chosen] == [0.90, 0.89, 0.88, 0.87, 0.86, 0.85]


def test_a_quota_of_zero_is_pure_ranking():
    items = [_item(DOOR_A, 0.9), _item(DOOR_B, 0.5)]

    assert len(_reserve_camera_slots(items, limit=1, quota=0)) == 1
    assert _reserve_camera_slots(items, limit=1, quota=0)[0].camera_id == DOOR_A


def test_the_default_bar_clears_what_chance_can_reach(tmp_path):
    """Calibration, pinned: 1330 pairs that cannot be the same person reach 0.440 by chance.

    A bar at or below that returns coincidences, and a single search draws hundreds of
    cross-camera comparisons, so the maximum is the number that matters -- not a percentile.
    """

    settings = Settings(data_dir=tmp_path)
    null_maximum = 0.440

    assert settings.reid_min_score_cross_camera > null_maximum


def test_a_coincidence_at_the_null_maximum_is_refused(tmp_path):
    settings = Settings(data_dir=tmp_path)

    admitted = _admit([_item(DOOR_B, 0.440)], settings, query_camera=DOOR_A)

    assert admitted == []


def test_candidate_pool_reaches_the_whole_small_gallery(tmp_path):
    class Db:
        def scalar(self, _statement):
            return 3248

    service = ReidIndexService(
        Db(),
        Settings(data_dir=tmp_path, reid_candidate_pool_max=5000),
    )

    assert service.candidate_pool_limit() == 3248


def test_candidate_pool_obeys_the_operational_cap(tmp_path):
    class Db:
        def scalar(self, _statement):
            return 12000

    service = ReidIndexService(
        Db(),
        Settings(data_dir=tmp_path, reid_candidate_pool_max=1000),
    )

    assert service.candidate_pool_limit() == 1000
