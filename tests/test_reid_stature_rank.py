"""Height as a tie-breaker on cross-camera matches: what it may move, and what it must not.

The measured strength sets the shape of these tests. Height agrees to 23 percentile points
between cross-camera candidate pairs where random pairs differ by 34 -- real (bootstrap p=0.018)
but far too weak to overrule appearance, and drawn from 15 pairs none of which is a confirmed
identity. So it reorders candidates the embedding cannot separate, and nothing else.
"""
import uuid

import pytest

from app.api.reid import _reserve_camera_slots, _stature_bonus, _stature_percentile
from app.schemas.reid import ReidMatchItem

WEIGHT = 0.03
CAMERA_A, CAMERA_B = uuid.uuid4(), uuid.uuid4()


def item(score, *, camera=CAMERA_B, percentile=None):
    return ReidMatchItem(
        crop_id=uuid.uuid4(), score=score, camera_id=camera, stature_percentile=percentile
    )


def test_a_matching_height_lifts_a_cross_camera_candidate():
    assert _stature_bonus(item(0.46, percentile=70), 70, CAMERA_A, WEIGHT) == pytest.approx(WEIGHT)


def test_a_clashing_height_pushes_one_down():
    assert _stature_bonus(item(0.46, percentile=10), 90, CAMERA_A, WEIGHT) == pytest.approx(-WEIGHT)


def test_the_nudge_is_bounded_by_the_weight():
    for query, candidate in ((0, 100), (100, 0), (50, 50), (20, 45)):
        bonus = _stature_bonus(item(0.46, percentile=candidate), query, CAMERA_A, WEIGHT)
        assert -WEIGHT <= bonus <= WEIGHT


def test_height_says_nothing_within_one_camera():
    """The embedding already separates people at 0.92 there; height would only blur it."""

    assert _stature_bonus(item(0.93, camera=CAMERA_A, percentile=10), 90, CAMERA_A, WEIGHT) == 0.0


def test_an_unmeasured_height_on_either_side_is_silent():
    assert _stature_bonus(item(0.46, percentile=None), 70, CAMERA_A, WEIGHT) == 0.0
    assert _stature_bonus(item(0.46, percentile=70), None, CAMERA_A, WEIGHT) == 0.0


def test_the_weight_can_be_turned_off():
    assert _stature_bonus(item(0.46, percentile=70), 70, CAMERA_A, 0.0) == 0.0


def test_agreement_is_reported_so_the_ranking_can_be_read():
    candidate = item(0.46, percentile=60)
    _stature_bonus(candidate, 70, CAMERA_A, WEIGHT)

    assert candidate.stature_agreement == pytest.approx(0.8)


def test_an_upload_with_no_camera_still_gets_the_nudge():
    """A query photo has no camera, so every match is a crossing and height is worth having."""

    assert _stature_bonus(item(0.46, percentile=70), 70, None, WEIGHT) == pytest.approx(WEIGHT)


def test_it_reorders_candidates_appearance_cannot_separate():
    close = [item(0.462, percentile=15), item(0.458, percentile=72)]
    ranking = {
        c.crop_id: c.score + _stature_bonus(c, 70, CAMERA_A, WEIGHT) for c in close
    }

    ordered = _reserve_camera_slots(close, 2, 0, key=lambda i: ranking[i.crop_id])

    assert ordered[0].stature_percentile == 72


def test_it_cannot_overturn_a_gap_appearance_is_sure_about():
    """A 0.48 against a 0.30 is not a close call, and height is not strong enough to say so."""

    far = [item(0.48, percentile=15), item(0.30, percentile=70)]
    ranking = {c.crop_id: c.score + _stature_bonus(c, 70, CAMERA_A, WEIGHT) for c in far}

    ordered = _reserve_camera_slots(far, 2, 0, key=lambda i: ranking[i.crop_id])

    assert ordered[0].score == 0.48


def test_the_query_percentile_is_read_off_the_crop_attributes():
    assert _stature_percentile({"stature": {"percentile": 73, "band": "average"}}) == 73
    assert _stature_percentile({"stature": {"band": "tall"}}) is None
    assert _stature_percentile({}) is None
    assert _stature_percentile(None) is None


# --- where it actually changes an answer -----------------------------------------------


def test_the_camera_link_picks_the_candidate_whose_height_agrees():
    """The links endpoint ranks a whole camera's crops with no threshold, which is the close
    call height was measured to help with. Two candidates the embedding cannot separate, and
    only one of them is the right size."""

    from app.api.reid import _stature_bonus

    query_percentile = 75
    candidates = [item(0.451, percentile=20), item(0.447, percentile=78)]
    best = max(
        candidates,
        key=lambda c: c.score + _stature_bonus(c, query_percentile, CAMERA_A, WEIGHT),
    )

    assert best.stature_percentile == 78


def test_the_link_still_reports_the_embedding_score_for_the_chance_test():
    """Height reorders candidates; it must not talk a weak score past the chance ceiling."""

    from app.api.reid import _stature_bonus

    candidate = item(0.42, percentile=70)
    ranking = candidate.score + _stature_bonus(candidate, 70, CAMERA_A, WEIGHT)

    assert ranking > 0.44  # the ranking value moved past the ceiling
    assert candidate.score < 0.44  # but what beats_chance is judged on did not
