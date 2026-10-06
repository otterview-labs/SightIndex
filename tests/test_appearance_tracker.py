"""Bounded visit sampling: what gets stored, retried, or dropped as a repeat."""

from typing import cast

import pytest

from app.services.appearance_tracker import AppearanceTracker


@pytest.fixture
def tracker() -> AppearanceTracker:
    return AppearanceTracker(match_distance=0.32, idle_seconds=6.0, max_seconds=300.0)


def test_someone_standing_still_is_stored_once(tracker: AppearanceTracker) -> None:
    """The bug this exists for: 127 crops of one person over five minutes."""

    stored = 0
    for step in range(150):  # five minutes at a 2s interval
        stored += len(tracker.new_visits([(0.5, 0.5)], now=step * 2.0))

    assert stored == 1


def test_someone_walking_across_the_frame_is_still_one_visit(
    tracker: AppearanceTracker,
) -> None:
    stored = 0
    for step in range(10):
        stored += len(tracker.new_visits([(0.1 + step * 0.05, 0.5)], now=step * 2.0))

    assert stored == 1


def test_leaving_and_coming_back_is_two_visits(tracker: AppearanceTracker) -> None:
    first = tracker.new_visits([(0.5, 0.5)], now=0.0)
    later = tracker.new_visits([(0.5, 0.5)], now=100.0)

    assert len(first) == 1
    assert len(later) == 1


def test_two_people_are_two_visits(tracker: AppearanceTracker) -> None:
    started = tracker.new_visits([(0.2, 0.5), (0.8, 0.5)], now=0.0)

    assert len(started) == 2
    assert tracker.new_visits([(0.2, 0.5), (0.8, 0.5)], now=2.0) == []


def test_two_people_close_together_do_not_collapse(
    tracker: AppearanceTracker,
) -> None:
    """One detection may claim one visit; without that both would match the same one."""

    tracker.new_visits([(0.50, 0.5)], now=0.0)
    started = tracker.new_visits([(0.50, 0.5), (0.55, 0.5)], now=2.0)

    assert len(started) == 1


def test_a_visit_that_never_ends_is_stored_again(
    tracker: AppearanceTracker,
) -> None:
    """Otherwise somebody at the door all afternoon appears once, at the moment they arrived."""

    stored = 0
    for step in range(400):  # over 13 minutes, past the 300s cap
        stored += len(tracker.new_visits([(0.5, 0.5)], now=step * 2.0))

    assert stored >= 2, "the max-duration cap never fired"


def test_the_returned_indices_point_at_the_right_detections(
    tracker: AppearanceTracker,
) -> None:
    tracker.new_visits([(0.2, 0.5)], now=0.0)

    started = tracker.new_visits([(0.2, 0.5), (0.9, 0.9)], now=2.0)

    assert started == [1], "the index must select the newcomer, not the one already stored"


def test_a_rolled_back_frame_does_not_swallow_its_visit(
    tracker: AppearanceTracker,
) -> None:
    """A frame dropped for backpressure is retried; its bodies must still count as new."""

    before = tracker.snapshot()
    assert tracker.new_visits([(0.5, 0.5)], now=0.0) == [0]
    tracker.restore(before)

    assert tracker.new_visits([(0.5, 0.5)], now=2.0) == [0], "the visit was lost on retry"


def test_a_snapshot_is_not_a_live_view(tracker: AppearanceTracker) -> None:
    before = tracker.snapshot()
    tracker.new_visits([(0.5, 0.5)], now=0.0)
    tracker.new_visits([(0.5, 0.5)], now=2.0)
    tracker.restore(before)

    assert tracker.new_visits([(0.5, 0.5)], now=4.0) == [0]


def test_default_configuration_does_not_resample_better_quality(
    tracker: AppearanceTracker,
) -> None:
    """Supplying quality cannot change legacy behaviour unless resampling is enabled."""

    assert tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[10.0]) == [0]
    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[100.0]) == []
    assert tracker.samples_saved == (1,)


def test_quality_improvement_respects_interval_and_ratio() -> None:
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
        min_sample_interval_seconds=2.0,
        quality_improvement_ratio=0.15,
    )

    assert tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[100.0]) == [0]
    assert tracker.new_visits([(0.5, 0.5)], now=1.9, qualities=[200.0]) == []
    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[114.9]) == []
    assert tracker.new_visits([(0.5, 0.5)], now=2.1, qualities=[115.0]) == [0]
    assert tracker.samples_saved == (2,)


def test_resampling_is_bounded_to_three_samples_per_visit() -> None:
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
        min_sample_interval_seconds=2.0,
        quality_improvement_ratio=0.15,
    )

    assert tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[100.0]) == [0]
    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[115.0]) == [0]
    assert tracker.new_visits([(0.5, 0.5)], now=4.0, qualities=[132.25]) == [0]
    assert tracker.new_visits([(0.5, 0.5)], now=6.0, qualities=[1_000.0]) == []
    assert tracker.samples_saved == (3,)


@pytest.mark.parametrize(
    "quality",
    [None, -1.0, float("nan"), float("inf"), float("-inf"), True, "bad"],
    ids=[
        "missing",
        "negative",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "boolean",
        "wrong-type",
    ],
)
def test_unknown_initial_quality_keeps_the_original_single_sample(
    quality: object,
) -> None:
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
    )

    runtime_quality = cast(float | None, quality)
    assert tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[runtime_quality]) == [0]
    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[100.0]) == []
    assert tracker.samples_saved == (1,)


@pytest.mark.parametrize(
    "quality",
    [None, -1.0, float("nan"), float("inf"), float("-inf"), True, "bad"],
    ids=[
        "missing",
        "negative",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "boolean",
        "wrong-type",
    ],
)
def test_invalid_candidate_quality_is_neutral(quality: object) -> None:
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
    )
    tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[10.0])

    runtime_quality = cast(float | None, quality)
    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[runtime_quality]) == []
    assert tracker.samples_saved == (1,)


def test_zero_quality_is_a_measured_baseline_that_can_improve() -> None:
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
    )

    assert tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[0.0]) == [0]
    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[0.0]) == []
    assert tracker.new_visits([(0.5, 0.5)], now=2.1, qualities=[0.01]) == [0]
    assert tracker.samples_saved == (2,)


def test_zero_quality_does_not_replace_a_positive_baseline() -> None:
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
    )
    tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[10.0])

    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[0.0]) == []
    assert tracker.samples_saved == (1,)


def test_zero_ratio_still_requires_strict_quality_improvement() -> None:
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
        quality_improvement_ratio=0.0,
    )

    assert tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[10.0]) == [0]
    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[10.0]) == []
    assert tracker.new_visits([(0.5, 0.5)], now=2.1, qualities=[10.001]) == [0]
    assert tracker.samples_saved == (2,)


def test_misaligned_qualities_are_rejected_before_state_changes() -> None:
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
    )

    with pytest.raises(ValueError, match="one value for each center"):
        tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[])

    assert tracker.samples_saved == ()


def test_snapshot_restore_retries_a_quality_improvement() -> None:
    """Backpressure rollback must restore the quality baseline and sample budget."""

    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
    )
    tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[10.0])
    before = tracker.snapshot()

    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[12.0]) == [0]
    assert tracker.samples_saved == (2,)
    tracker.restore(before)
    assert tracker.samples_saved == (1,)
    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[12.0]) == [0]

    tracker.restore(before)
    assert tracker.samples_saved == (1,), "restore must not mutate the reusable snapshot"


def test_retained_tokens_are_defensive_and_preserved_by_snapshot_restore() -> None:
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
    )
    tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[10.0])
    original_token = tracker.retained_tokens()[0]
    before = tracker.snapshot()

    exposed = tracker.retained_tokens()
    exposed[0] = "caller-mutation"
    assert tracker.retained_tokens() == {0: original_token}

    tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[12.0])
    tracker.restore(before)
    assert tracker.retained_tokens() == {}
    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[12.0]) == [0]
    assert tracker.retained_tokens() == {0: original_token}


def test_failed_first_sample_removes_the_uncommitted_visit() -> None:
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
    )
    before = tracker.snapshot()
    assert tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[10.0]) == [0]
    failed_token = tracker.retained_tokens()[0]

    tracker.rollback_retained({failed_token}, before)

    assert tracker.samples_saved == ()
    assert tracker.retained_tokens() == {}
    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[10.0]) == [0]
    assert tracker.retained_tokens()[0] != failed_token


def test_failed_quality_upgrade_restores_its_budget_for_retry() -> None:
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
    )
    tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[10.0])
    sampling_token = tracker.retained_tokens()[0]
    before = tracker.snapshot()
    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[12.0]) == [0]

    tracker.rollback_retained({sampling_token}, before)

    assert tracker.samples_saved == (1,)
    assert tracker.retained_tokens() == {}
    assert tracker.new_visits([(0.5, 0.5)], now=2.1, qualities=[12.0]) == [0]
    assert tracker.retained_tokens() == {0: sampling_token}


def test_partial_failure_rolls_back_only_its_visit() -> None:
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
    )
    centers = [(0.2, 0.5), (0.8, 0.5)]
    tracker.new_visits(centers, now=0.0, qualities=[10.0, 10.0])
    initial_tokens = tracker.retained_tokens()
    before = tracker.snapshot()
    assert tracker.new_visits(centers, now=2.0, qualities=[12.0, 12.0]) == [0, 1]

    tracker.rollback_retained({initial_tokens[0]}, before)

    assert tracker.samples_saved == (1, 2)
    assert tracker.retained_tokens() == {1: initial_tokens[1]}
    assert tracker.new_visits(centers, now=2.1, qualities=[12.0, 12.0]) == [0]
    assert tracker.retained_tokens() == {0: initial_tokens[0]}


def test_unknown_rollback_token_is_safe() -> None:
    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=6.0,
        max_seconds=300.0,
        max_samples_per_visit=3,
    )
    tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[10.0])
    before = tracker.snapshot()
    tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[12.0])
    retained = tracker.retained_tokens()

    tracker.rollback_retained({"unknown-token"}, before)

    assert tracker.samples_saved == (2,)
    assert tracker.retained_tokens() == retained


def test_quality_resampling_does_not_extend_the_visit_lifetime() -> None:
    """A better crop supplements a visit; it does not turn it into a new visit."""

    tracker = AppearanceTracker(
        match_distance=0.32,
        idle_seconds=100.0,
        max_seconds=5.0,
        max_samples_per_visit=3,
        min_sample_interval_seconds=2.0,
        quality_improvement_ratio=0.10,
    )

    assert tracker.new_visits([(0.5, 0.5)], now=0.0, qualities=[100.0]) == [0]
    assert tracker.new_visits([(0.5, 0.5)], now=2.0, qualities=[110.0]) == [0]
    assert tracker.new_visits([(0.5, 0.5)], now=4.0, qualities=[121.0]) == [0]
    assert tracker.new_visits([(0.5, 0.5)], now=6.0, qualities=[200.0]) == [0]
    assert tracker.samples_saved == (1,)


@pytest.mark.parametrize("max_samples", [0, 4, True, 2.5, "2"])
def test_max_samples_per_visit_must_stay_within_bound(max_samples: object) -> None:
    with pytest.raises(ValueError, match="integer from 1 to 3"):
        AppearanceTracker(
            match_distance=0.32,
            idle_seconds=6.0,
            max_seconds=300.0,
            max_samples_per_visit=cast(int, max_samples),
        )


@pytest.mark.parametrize("interval", [-1.0, float("nan"), float("inf")])
def test_sample_interval_must_be_finite_and_non_negative(interval: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        AppearanceTracker(
            match_distance=0.32,
            idle_seconds=6.0,
            max_seconds=300.0,
            min_sample_interval_seconds=interval,
        )


@pytest.mark.parametrize("ratio", [-0.01, float("nan"), float("inf")])
def test_quality_ratio_must_be_finite_and_non_negative(ratio: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        AppearanceTracker(
            match_distance=0.32,
            idle_seconds=6.0,
            max_seconds=300.0,
            quality_improvement_ratio=ratio,
        )
