"""Keeps one crop per visit, instead of one per capture interval.

A doorway sampled every two seconds stores the same person over and over. Measured on a live
feed here, one person standing near the door produced 127 crops across five minutes. Each one is
written to disk, embedded on the GPU, indexed in Milvus and returned by every search -- so a
single loiterer crowds everyone else out of the results, and the index grows with nothing new
in it.

This is deliberately not identity matching. It answers the narrower question "is this the same
body the previous frame already stored", which position answers well enough at a two-second
interval, and it never has to be right across cameras or across time. Whether two *visits* are
the same person is what ReID is for, and that stays a query-time question.
"""

from dataclasses import dataclass


@dataclass
class _Visit:
    center: tuple[float, float]
    started_at: float
    last_seen: float


class AppearanceTracker:
    """Per-stream state: which bodies on screen have already been stored.

    One instance belongs to one capture loop and is not thread-safe, which is fine because a
    stream is captured by exactly one thread.
    """

    def __init__(
        self,
        *,
        match_distance: float,
        idle_seconds: float,
        max_seconds: float,
    ) -> None:
        self.match_distance = match_distance
        self.idle_seconds = idle_seconds
        self.max_seconds = max_seconds
        self._visits: list[_Visit] = []

    def new_visits(self, centers: list[tuple[float, float]], now: float) -> list[int]:
        """Indices of the detections that begin a visit; the rest repeat a stored body.

        Centres are normalised to the frame, so the same threshold holds at any resolution.
        """

        self._expire(now)
        matched: set[int] = set()
        started: list[int] = []
        for index, center in enumerate(centers):
            visit = self._match(center, matched)
            if visit is None:
                self._visits.append(_Visit(center=center, started_at=now, last_seen=now))
                matched.add(len(self._visits) - 1)
                started.append(index)
                continue
            position = self._visits.index(visit)
            matched.add(position)
            visit.center = center
            visit.last_seen = now
        return started

    def snapshot(self) -> list[_Visit]:
        """State to restore if the frame that consumed it is rolled back.

        A frame dropped for backpressure must not leave its visits recorded: the retry would see
        the same bodies, call them repeats, and that person would never be stored at all.
        """

        return [_Visit(v.center, v.started_at, v.last_seen) for v in self._visits]

    def restore(self, snapshot: list[_Visit]) -> None:
        self._visits = snapshot

    def _expire(self, now: float) -> None:
        # A visit ends when the body stops appearing. The max also ends one that never does:
        # somebody who stands at the door all afternoon should still show up more than once,
        # and without a cap they would be stored on arrival and never again.
        self._visits = [
            visit
            for visit in self._visits
            if now - visit.last_seen <= self.idle_seconds
            and now - visit.started_at <= self.max_seconds
        ]

    def _match(self, center: tuple[float, float], matched: set[int]) -> _Visit | None:
        best: _Visit | None = None
        best_distance = self.match_distance
        for position, visit in enumerate(self._visits):
            if position in matched:
                continue  # one detection per visit per frame, or two people merge into one
            distance = (
                (visit.center[0] - center[0]) ** 2 + (visit.center[1] - center[1]) ** 2
            ) ** 0.5
            if distance < best_distance:
                best_distance = distance
                best = visit
        return best
