import uuid
from datetime import datetime, timedelta

from app.schemas.reid import ReidMatchItem
from app.services.reid_index import collapse_occurrences

DOOR_A = uuid.uuid4()
DOOR_B = uuid.uuid4()
START = datetime(2026, 8, 24, 9, 40, 0)


def _item(camera, offset_seconds, score, captured=True):
    return ReidMatchItem(
        crop_id=uuid.uuid4(),
        score=score,
        camera_id=camera,
        captured_at=START + timedelta(seconds=offset_seconds) if captured else None,
    )


def _identity(*items, axis: int):
    """A unit vector per identity: same axis means cosine 1.0, different axis means 0.0."""

    vector = [0.0] * 8
    vector[axis] = 1.0
    return {item.crop_id: list(vector) for item in items}


def test_consecutive_frames_at_one_camera_become_one_visit():
    items = [_item(DOOR_A, 0, 0.97), _item(DOOR_A, 2, 0.91), _item(DOOR_A, 4, 0.87)]

    visits = collapse_occurrences(items, window_seconds=60, limit=10)

    assert len(visits) == 1
    assert visits[0].score == 0.97
    assert visits[0].frame_count == 3
    assert visits[0].first_seen == START
    assert visits[0].last_seen == START + timedelta(seconds=4)


def test_the_other_camera_survives_a_saturated_result_set():
    """The whole point: one doorway's frames must not bury the only cross-camera hit."""

    crowd = [_item(DOOR_A, index * 2, 0.9) for index in range(30)]
    elsewhere = _item(DOOR_B, 600, 0.62)

    visits = collapse_occurrences([*crowd, elsewhere], window_seconds=60, limit=5)

    assert [visit.camera_id for visit in visits] == [DOOR_A, DOOR_B]
    assert visits[0].frame_count == 30
    assert visits[1].frame_count == 1


def test_a_gap_longer_than_the_window_is_a_second_visit():
    items = [_item(DOOR_A, 0, 0.9), _item(DOOR_A, 30, 0.8), _item(DOOR_A, 400, 0.7)]

    visits = collapse_occurrences(items, window_seconds=60, limit=10)

    assert [visit.frame_count for visit in visits] == [2, 1]


def test_a_long_stay_stays_one_visit_across_any_bucket_boundary():
    """Gap-based, not fixed buckets: 5 minutes of 10s frames is one visit, not five."""

    items = [_item(DOOR_A, index * 10, 0.9 - index * 0.001) for index in range(30)]

    visits = collapse_occurrences(items, window_seconds=60, limit=10)

    assert len(visits) == 1
    assert visits[0].frame_count == 30


def test_undated_crops_are_never_merged_together():
    items = [_item(DOOR_A, 0, 0.9, captured=False), _item(DOOR_A, 0, 0.8, captured=False)]

    visits = collapse_occurrences(items, window_seconds=60, limit=10)

    assert [visit.frame_count for visit in visits] == [1, 1]
    assert all(visit.first_seen is None for visit in visits)


def test_a_zero_window_returns_every_frame_untouched():
    items = [_item(DOOR_A, 0, 0.9), _item(DOOR_A, 2, 0.8)]

    visits = collapse_occurrences(items, window_seconds=0, limit=10)

    assert [visit.frame_count for visit in visits] == [1, 1]


def test_the_limit_counts_visits_not_frames():
    items = [
        *[_item(DOOR_A, index, 0.9) for index in range(5)],
        *[_item(DOOR_B, index, 0.8) for index in range(5)],
    ]

    visits = collapse_occurrences(items, window_seconds=60, limit=1)

    assert len(visits) == 1
    assert visits[0].frame_count == 5



def test_two_people_at_one_door_do_not_chain_into_one_visit():
    """The bug this replaced: every gap under the window, so strangers merged into one visit."""

    alice = [_item(DOOR_A, index * 2, 0.95) for index in range(4)]
    bob = [_item(DOOR_A, index * 2 + 1, 0.62) for index in range(4)]
    vectors = {**_identity(*alice, axis=0), **_identity(*bob, axis=1)}

    visits = collapse_occurrences(
        [*alice, *bob], window_seconds=60, limit=10, vectors=vectors, identity_threshold=0.7
    )

    assert len(visits) == 2
    assert sorted(visit.frame_count for visit in visits) == [4, 4]
    assert sorted(round(visit.score, 2) for visit in visits) == [0.62, 0.95]


def test_frames_sharing_a_timestamp_split_when_they_are_different_people():
    """Not automatic: a doubled detection of one person also shares a timestamp, and merges."""

    together = [_item(DOOR_A, 0, 0.84), _item(DOOR_A, 0, 0.77)]
    vectors = {**_identity(together[0], axis=0), **_identity(together[1], axis=1)}

    visits = collapse_occurrences(
        together, window_seconds=60, limit=10, vectors=vectors, identity_threshold=0.7
    )

    assert [visit.frame_count for visit in visits] == [1, 1]


def test_a_drifting_run_does_not_chain_end_to_end():
    """Neighbours are near-identical while the ends are strangers -- single linkage merges the
    lot, which is how a real five-minute doorway run became one visit."""

    import math

    items = [_item(DOOR_A, index * 2, 0.9) for index in range(10)]
    # Each step rotates the vector slightly: adjacent pairs ~0.995, first-to-last ~0.62.
    vectors = {}
    for index, item in enumerate(items):
        angle = index * 0.1
        vector = [0.0] * 8
        vector[0], vector[1] = math.cos(angle), math.sin(angle)
        vectors[item.crop_id] = vector

    visits = collapse_occurrences(
        items, window_seconds=60, limit=10, vectors=vectors, identity_threshold=0.7
    )

    assert len(visits) > 1, "single-linkage chaining is back"
    assert max(visit.frame_count for visit in visits) < len(items)


def test_one_person_still_merges_when_vectors_agree():
    items = [_item(DOOR_A, index * 2, 0.9 - index * 0.01) for index in range(5)]

    visits = collapse_occurrences(
        items,
        window_seconds=60,
        limit=10,
        vectors=_identity(*items, axis=0),
        identity_threshold=0.7,
    )

    assert len(visits) == 1
    assert visits[0].frame_count == 5


def test_a_person_returning_later_is_a_second_visit_even_with_the_same_vector():
    """Identity alone must not merge across time; the window still bounds a visit."""

    items = [_item(DOOR_A, 0, 0.9), _item(DOOR_A, 600, 0.85)]

    visits = collapse_occurrences(
        items,
        window_seconds=60,
        limit=10,
        vectors=_identity(*items, axis=0),
        identity_threshold=0.7,
    )

    assert [visit.frame_count for visit in visits] == [1, 1]


def test_missing_vectors_fall_back_to_time_only_grouping():
    """A failed vector fetch degrades the grouping; it must not drop or duplicate results."""

    items = [_item(DOOR_A, index * 2, 0.9) for index in range(3)]

    visits = collapse_occurrences(
        items, window_seconds=60, limit=10, vectors={}, identity_threshold=0.7
    )

    assert len(visits) == 1
    assert visits[0].frame_count == 3


def test_a_frame_joins_the_group_it_resembles_not_the_nearest_in_time():
    """Alice, Bob, then Alice again: the last frame belongs to Alice's group, not Bob's."""

    alice_one = _item(DOOR_A, 0, 0.95)
    bob = _item(DOOR_A, 2, 0.60)
    alice_two = _item(DOOR_A, 4, 0.93)
    vectors = {**_identity(alice_one, alice_two, axis=0), **_identity(bob, axis=1)}

    visits = collapse_occurrences(
        [alice_one, bob, alice_two],
        window_seconds=60,
        limit=10,
        vectors=vectors,
        identity_threshold=0.7,
    )

    assert len(visits) == 2
    assert max(visits, key=lambda visit: visit.score).frame_count == 2
