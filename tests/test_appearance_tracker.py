"""One crop per visit: what gets stored, and what gets dropped as a repeat."""
import pytest

from app.services.appearance_tracker import AppearanceTracker


@pytest.fixture
def tracker():
    return AppearanceTracker(match_distance=0.32, idle_seconds=6.0, max_seconds=300.0)


def test_someone_standing_still_is_stored_once(tracker):
    """The bug this exists for: 127 crops of one person over five minutes."""

    stored = 0
    for step in range(150):  # five minutes at a 2s interval
        stored += len(tracker.new_visits([(0.5, 0.5)], now=step * 2.0))

    assert stored == 1


def test_someone_walking_across_the_frame_is_still_one_visit(tracker):
    stored = 0
    for step in range(10):
        stored += len(tracker.new_visits([(0.1 + step * 0.05, 0.5)], now=step * 2.0))

    assert stored == 1


def test_leaving_and_coming_back_is_two_visits(tracker):
    first = tracker.new_visits([(0.5, 0.5)], now=0.0)
    later = tracker.new_visits([(0.5, 0.5)], now=100.0)

    assert len(first) == 1
    assert len(later) == 1


def test_two_people_are_two_visits(tracker):
    started = tracker.new_visits([(0.2, 0.5), (0.8, 0.5)], now=0.0)

    assert len(started) == 2
    assert tracker.new_visits([(0.2, 0.5), (0.8, 0.5)], now=2.0) == []


def test_two_people_close_together_do_not_collapse(tracker):
    """One detection may claim one visit; without that both would match the same one."""

    tracker.new_visits([(0.50, 0.5)], now=0.0)
    started = tracker.new_visits([(0.50, 0.5), (0.55, 0.5)], now=2.0)

    assert len(started) == 1


def test_a_visit_that_never_ends_is_stored_again(tracker):
    """Otherwise somebody at the door all afternoon appears once, at the moment they arrived."""

    stored = 0
    for step in range(400):  # over 13 minutes, past the 300s cap
        stored += len(tracker.new_visits([(0.5, 0.5)], now=step * 2.0))

    assert stored >= 2, "the max-duration cap never fired"


def test_the_returned_indices_point_at_the_right_detections(tracker):
    tracker.new_visits([(0.2, 0.5)], now=0.0)

    started = tracker.new_visits([(0.2, 0.5), (0.9, 0.9)], now=2.0)

    assert started == [1], "the index must select the newcomer, not the one already stored"


def test_a_rolled_back_frame_does_not_swallow_its_visit(tracker):
    """A frame dropped for backpressure is retried; its bodies must still count as new."""

    before = tracker.snapshot()
    assert tracker.new_visits([(0.5, 0.5)], now=0.0) == [0]
    tracker.restore(before)

    assert tracker.new_visits([(0.5, 0.5)], now=2.0) == [0], "the visit was lost on retry"


def test_a_snapshot_is_not_a_live_view(tracker):
    before = tracker.snapshot()
    tracker.new_visits([(0.5, 0.5)], now=0.0)
    tracker.new_visits([(0.5, 0.5)], now=2.0)
    tracker.restore(before)

    assert tracker.new_visits([(0.5, 0.5)], now=4.0) == [0]
