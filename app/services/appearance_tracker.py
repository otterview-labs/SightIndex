"""Keeps a bounded set of useful crops per positional visit, not one per capture interval.

A doorway sampled every two seconds stores the same person over and over. Measured on a live
feed here, one person standing near the door produced 127 crops across five minutes. Each one is
written to disk, embedded on the GPU, indexed in Milvus and returned by every search -- so a
single loiterer crowds everyone else out of the results, and the index grows with nothing new
in it.

This is deliberately not identity matching. It answers the narrower question "is this the same
body the previous frame already stored", which position answers well enough at a two-second
interval, and it never has to be right across cameras or across time. Whether two *visits* are
the same person is what ReID is for, and that stays a query-time question. The default remains
one crop; callers may retain a small number of objectively better later crops.
"""

import math
from dataclasses import dataclass
from numbers import Real
from uuid import uuid4


@dataclass
class _Visit:
    sampling_token: str
    center: tuple[float, float]
    started_at: float
    last_seen: float
    samples_saved: int
    best_quality: float | None
    last_sampled_at: float


class AppearanceTracker:
    """Per-stream state: which bodies on screen have already been stored.

    One instance belongs to one capture loop and is not thread-safe, which is fine because a
    stream is captured by exactly one thread. A visit is only a short-lived positional grouping;
    neither a match nor a quality resample verifies that two detections have the same identity.
    """

    def __init__(
        self,
        *,
        match_distance: float,
        idle_seconds: float,
        max_seconds: float,
        max_samples_per_visit: int = 1,
        min_sample_interval_seconds: float = 2.0,
        quality_improvement_ratio: float = 0.15,
    ) -> None:
        """Configure positional visit tracking and optional quality-based resampling.

        Args:
            match_distance: Maximum normalised centre distance for a positional match.
            idle_seconds: Time without a match after which a visit expires.
            max_seconds: Maximum lifetime of one continuously observed visit.
            max_samples_per_visit: Maximum samples retained for one visit. The default of one
                preserves the original one-crop-per-visit behaviour. Values above three are
                rejected to keep retention bounded.
            min_sample_interval_seconds: Minimum elapsed time between retained samples.
            quality_improvement_ratio: Required relative improvement over the best retained
                sample, expressed as a non-negative fraction.

        Raises:
            ValueError: If a quality-resampling setting is outside its supported range.
        """

        if (
            isinstance(max_samples_per_visit, bool)
            or not isinstance(max_samples_per_visit, int)
            or not 1 <= max_samples_per_visit <= 3
        ):
            raise ValueError("max_samples_per_visit must be an integer from 1 to 3")
        if (
            isinstance(min_sample_interval_seconds, bool)
            or not math.isfinite(min_sample_interval_seconds)
            or min_sample_interval_seconds < 0.0
        ):
            raise ValueError("min_sample_interval_seconds must be finite and non-negative")
        if (
            isinstance(quality_improvement_ratio, bool)
            or not math.isfinite(quality_improvement_ratio)
            or quality_improvement_ratio < 0.0
        ):
            raise ValueError("quality_improvement_ratio must be finite and non-negative")

        self.match_distance: float = match_distance
        self.idle_seconds: float = idle_seconds
        self.max_seconds: float = max_seconds
        self.max_samples_per_visit: int = max_samples_per_visit
        self.min_sample_interval_seconds: float = min_sample_interval_seconds
        self.quality_improvement_ratio: float = quality_improvement_ratio
        self._visits: list[_Visit] = []
        self._retained_tokens: dict[int, str] = {}

    @property
    def samples_saved(self) -> tuple[int, ...]:
        """Return retained-sample counts for the current positional visits.

        The tuple follows the tracker's transient visit order and is intended for diagnostics;
        it is not a stable identity mapping.
        """

        return tuple(visit.samples_saved for visit in self._visits)

    def new_visits(
        self,
        centers: list[tuple[float, float]],
        now: float,
        qualities: list[float | None] | None = None,
    ) -> list[int]:
        """Select detections that should be retained for their positional visit.

        Centres are normalised to the frame, so the same threshold holds at any resolution.
        New visits are always selected. Existing visits are selected only when bounded
        quality-based resampling is enabled and a measured quality improves sufficiently over
        the best retained sample. Missing, negative, and non-finite qualities are neutral. Zero
        is a measured low-quality baseline that a later positive-quality sample may improve.

        Args:
            centers: Normalised ``(x, y)`` centres in detection order.
            now: Current monotonic timestamp in seconds.
            qualities: Optional ROI quality values aligned one-to-one with ``centers``.

        Returns:
            Detection indices to retain, in the same order as ``centers``.

        Raises:
            ValueError: If ``qualities`` is present but is not aligned with ``centers``.
        """

        if qualities is not None and len(qualities) != len(centers):
            raise ValueError("qualities must contain one value for each center")

        self._expire(now)
        matched: set[int] = set()
        selected: list[int] = []
        retained_tokens: dict[int, str] = {}
        for index, center in enumerate(centers):
            quality = self._measured_quality(None if qualities is None else qualities[index])
            visit = self._match(center, matched)
            if visit is None:
                self._visits.append(
                    _Visit(
                        sampling_token=uuid4().hex,
                        center=center,
                        started_at=now,
                        last_seen=now,
                        samples_saved=1,
                        best_quality=quality,
                        last_sampled_at=now,
                    )
                )
                matched.add(len(self._visits) - 1)
                selected.append(index)
                retained_tokens[index] = self._visits[-1].sampling_token
                continue
            position = self._visits.index(visit)
            matched.add(position)
            visit.center = center
            visit.last_seen = now
            if self._should_resample(visit, quality, now):
                visit.samples_saved += 1
                visit.best_quality = quality
                visit.last_sampled_at = now
                selected.append(index)
                retained_tokens[index] = visit.sampling_token
        self._retained_tokens = retained_tokens
        return selected

    def retained_tokens(self) -> dict[int, str]:
        """Return detection-index-to-budget-token mappings from the last selection.

        Tokens exist only to commit or roll back retained-sample budgets. They are scoped to
        this in-memory tracker and must not be used as person identities or persisted tracks.

        Returns:
            A defensive copy keyed by the detection indices returned by :meth:`new_visits`.
        """

        return dict(self._retained_tokens)

    def rollback_retained(self, tokens: set[str], snapshot: list[_Visit]) -> None:
        """Roll back only failed retained samples, leaving successful siblings committed.

        For an improved sample, only the saved-sample count, quality baseline, and last sample
        time are restored. Positional tracking remains current. For a failed first sample, the
        new visit is removed because it had no state in the pre-selection snapshot. Unknown or
        already-removed tokens are ignored.

        Args:
            tokens: Sampling-budget tokens whose downstream crop retention failed.
            snapshot: State captured immediately before the corresponding :meth:`new_visits`
                call.
        """

        if not tokens:
            return

        previous_by_token = {visit.sampling_token: visit for visit in snapshot}
        retained_visits: list[_Visit] = []
        for visit in self._visits:
            if visit.sampling_token not in tokens:
                retained_visits.append(visit)
                continue
            previous = previous_by_token.get(visit.sampling_token)
            if previous is None:
                continue
            visit.samples_saved = previous.samples_saved
            visit.best_quality = previous.best_quality
            visit.last_sampled_at = previous.last_sampled_at
            retained_visits.append(visit)
        self._visits = retained_visits
        self._retained_tokens = {
            index: token for index, token in self._retained_tokens.items() if token not in tokens
        }

    def snapshot(self) -> list[_Visit]:
        """State to restore if the frame that consumed it is rolled back.

        A frame dropped for backpressure must not leave its visits recorded: the retry would see
        the same bodies, call them repeats, and that person would never be stored at all. Quality
        baselines and sample budgets are copied too, so a selected improvement can be retried.
        """

        return [self._copy_visit(visit) for visit in self._visits]

    def restore(self, snapshot: list[_Visit]) -> None:
        """Restore an independent copy of a previously captured snapshot.

        Args:
            snapshot: State previously returned by :meth:`snapshot`.
        """

        self._visits = [self._copy_visit(visit) for visit in snapshot]
        self._retained_tokens = {}

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
        """Return the nearest unmatched positional visit within the configured radius."""

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

    def _should_resample(
        self,
        visit: _Visit,
        quality: float | None,
        now: float,
    ) -> bool:
        """Return whether a measured improvement may consume another sample slot."""

        if (
            visit.samples_saved >= self.max_samples_per_visit
            or quality is None
            or visit.best_quality is None
            or quality <= visit.best_quality
            or now - visit.last_sampled_at < self.min_sample_interval_seconds
        ):
            return False
        required_quality = visit.best_quality * (1.0 + self.quality_improvement_ratio)
        return quality > required_quality or math.isclose(
            quality,
            required_quality,
            rel_tol=1e-12,
            abs_tol=0.0,
        )

    @staticmethod
    def _measured_quality(quality: object) -> float | None:
        """Normalise a quality reading, treating invalid runtime values as unknown."""

        if quality is None or isinstance(quality, bool) or not isinstance(quality, Real):
            return None
        measured = float(quality)
        if not math.isfinite(measured) or measured < 0.0:
            return None
        return measured

    @staticmethod
    def _copy_visit(visit: _Visit) -> _Visit:
        """Copy every mutable visit-state field for rollback isolation."""

        return _Visit(
            sampling_token=visit.sampling_token,
            center=visit.center,
            started_at=visit.started_at,
            last_seen=visit.last_seen,
            samples_saved=visit.samples_saved,
            best_quality=visit.best_quality,
            last_sampled_at=visit.last_sampled_at,
        )
