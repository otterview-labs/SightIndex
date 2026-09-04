"""Relative stature from where a person's feet land, with no model and no tape measure.

Clothing colour is the wrong thing to search a person by here: measured over 2806 crops from
these cameras, upper garments split 48% dark and 52% light, so colour halves the crowd and stops.
Worse, it is the wrong *kind* of attribute -- it changes daily, and the ReID embedding is already
keying on it, which is why two different people in similar outfits scored 0.440, right at the
chance ceiling. Height does not change and is not what the embedding looks at.

The geometry is the whole method. With a fixed camera and a flat floor, the imaged height of a
fixed-height object is a linear function of the row its feet land on. Fit that line over the
crowd and a person's own height divided by the line's prediction is their stature relative to
everyone who walked through -- no calibration target, no metric units, no walk-through.

It carries identity. Crops that ReID scores above 0.9 on one camera -- same-person territory,
since same-camera best matches run 0.925 to 0.946 -- agree on the ratio to within 0.017, while
two random people differ by 0.11. Six times the separation, so roughly 2% noise against 11% real
spread.

Choosing the surface took two tries, and the first was wrong in an instructive way. Judged on
separation alone, a bare line in the foot row looked best: 10.4x, against 6.5x for a line that
also used the column and 5.6x for a full quadratic. The obvious reading was that extra freedom
was absorbing real height. It was not. The bare line left the ratio correlated with the *column*
at 0.714 -- most of that separation was "this person walks down the left-hand side", held
constant across their crops exactly the way a height would be. Separation only shows a quantity
is stable per person; it cannot tell stature from a habit.

So position independence is the gate and separation is scored after it. A full second-order
surface brings the correlations to 0.040 with the column and 0.056 with the row, and still
separates people 6.4x on one camera and 6.9x on the other. Those are the honest numbers.

The ratio is not comparable between cameras, so a rank is published beside it. Measured across
the two doorways here, the ratio's 10th-to-90th spread is 0.935-1.062 on one and 0.882-1.150 on
the other -- 2.1 times wider -- although the same people walk through both, so the extra width
belongs to the camera, not the crowd. A raw 1.06 is the 90th percentile at one door and around
the 65th at the other. The percentile is therefore what any cross-camera comparison should use;
the ratio is kept because it is the raw measurement and is comparable within one camera.

Whether stature survives the trip between cameras is supported but not settled. With no labelled
identities, the strongest available proxy was a cross-camera ReID best match inside a plausible
walking time: 15 such pairs agree on stature to 0.062 where two random people from opposite doors
differ by 0.100, which a bootstrap puts at p=0.018. That is real evidence and a small sample, and
every one of those pairs scored below the 0.440 chance ceiling, so none is a confirmed identity.
One person walking through both doors a few times would settle it properly.

Known limits. A person bending or crouching images shorter and reads short; nothing here detects
that. A box clipped by the frame edge is not a whole person, so those are refused outright, as
are cameras with too few samples to fit. The surface has six parameters and is fitted per camera,
so a re-aimed camera needs its cache dropped.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None  # type: ignore[assignment]

from sqlalchemy import select

# A box within this many pixels of any edge is treated as clipped.
_EDGE_MARGIN = 6
# Below this the fit is noise, and a stature drawn from it would be too. The surface has six
# parameters, so this is a wide margin over what it takes to determine them.
_MIN_SAMPLES = 200
_MIN_CONFIDENCE = 0.70
# The crowd changes slowly and refitting scans a table, so an hour of staleness is cheap.
_CACHE_SECONDS = 3600.0

_cache: dict[str, tuple[float, Calibration | None]] = {}


def _design(foot_x, foot_y):
    """Second-order surface in the foot position.

    A plane in the foot row alone is what the pinhole geometry predicts for a flat floor, and it
    fits well, but what it leaves behind is not stature -- see the module docstring. The extra
    terms exist to drive the residual's dependence on position to zero, which is the condition
    for calling the residual a property of the person.
    """

    one = np.ones_like(foot_y)
    return np.column_stack([one, foot_y, foot_x, foot_y**2, foot_x**2, foot_x * foot_y])


@dataclass(frozen=True)
class Calibration:
    """One camera's floor surface, plus where its crowd's middle sits."""

    coefficients: tuple[float, ...]
    samples: int
    # The camera's own ratio distribution at percentiles 0..100, so a crop can be placed in the
    # crowd it was measured against rather than on a scale only this camera uses.
    quantiles: tuple[float, ...]
    frame_width: int
    frame_height: int

    def rank(self, ratio: float) -> int:
        """Where this ratio falls among the people who walked past this camera, 0 to 100."""

        position = int(np.searchsorted(np.array(self.quantiles), ratio, side="right"))
        return max(0, min(100, position))

    def predict(self, foot_x: float, foot_y: float) -> float:
        row = _design(np.array([float(foot_x)]), np.array([float(foot_y)]))[0]
        return float(np.dot(row, np.array(self.coefficients)))


def reset_cache() -> None:
    """Drops fitted lines. Call after a camera is re-aimed, and in tests."""

    _cache.clear()


class StatureService:
    def __init__(self, db, settings) -> None:
        self.db = db
        self.settings = settings

    def describe(self, bbox: dict | None, camera_id: uuid.UUID | str | None) -> dict | None:
        """`{"ratio": .., "band": ..}` for this crop, or None when it cannot be measured."""

        calibration = self.calibration(camera_id)
        if calibration is None:
            return None
        measured = self._usable(bbox, calibration.frame_width, calibration.frame_height)
        if measured is None:
            return None
        foot_x, foot_y, height = measured
        expected = calibration.predict(foot_x, foot_y)
        if expected <= 1.0:
            return None
        ratio = height / expected
        if not 0.5 <= ratio <= 1.8:
            # Outside anything a standing person produces: a bad box, not a very tall person.
            return None
        percentile = calibration.rank(ratio)
        if percentile <= 20:
            band = "short"
        elif percentile >= 80:
            band = "tall"
        else:
            band = "average"
        # Bands and percentile are the camera-independent parts; ratio is the raw reading.
        return {"ratio": round(float(ratio), 3), "percentile": percentile, "band": band}

    def calibration(self, camera_id: uuid.UUID | str | None) -> Calibration | None:
        if camera_id is None or np is None:
            return None
        key = str(camera_id)
        cached = _cache.get(key)
        now = time.monotonic()
        if cached is not None and now - cached[0] < _CACHE_SECONDS:
            return cached[1]
        fitted = self._fit(key)
        _cache[key] = (now, fitted)
        return fitted

    def _fit(self, camera_id: str) -> Calibration | None:
        from app.models.media import Image, PersonCrop

        rows = list(
            self.db.execute(
                select(PersonCrop.bbox, Image.image_url)
                .join(Image, Image.id == PersonCrop.image_id)
                .where(Image.camera_id == camera_id)
            )
        )
        if len(rows) < _MIN_SAMPLES:
            return None
        frame = self._frame_size(rows)
        if frame is None:
            return None
        width, height = frame

        points = []
        for bbox, _url in rows:
            measured = self._usable(bbox, width, height)
            if measured is not None:
                points.append(measured)
        if len(points) < _MIN_SAMPLES:
            return None

        data = np.array(points, dtype="float64")
        foot_x, foot_y, pixel_height = data[:, 0], data[:, 1], data[:, 2]
        design = _design(foot_x, foot_y)
        coefficients, *_ = np.linalg.lstsq(design, pixel_height, rcond=None)
        predicted = np.clip(design @ coefficients, 1.0, None)
        ratios = pixel_height / predicted
        return Calibration(
            coefficients=tuple(float(c) for c in coefficients),
            samples=len(points),
            # Ranked against this camera's own crowd, so "tall" means tall for this doorway
            # rather than tall against a population this deployment never measured.
            quantiles=tuple(float(q) for q in np.percentile(ratios, np.arange(0, 101))),
            frame_width=width,
            frame_height=height,
        )

    def _frame_size(self, rows) -> tuple[int, int] | None:
        """Frame dimensions, from the bbox when the capture recorded them, else from a frame."""

        for bbox, _url in rows:
            if isinstance(bbox, dict) and bbox.get("frame_width") and bbox.get("frame_height"):
                return int(bbox["frame_width"]), int(bbox["frame_height"])
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - opencv ships with the vision extra
            return None
        prefix = "/data/"
        for _bbox, url in rows:
            if not url or not url.startswith(prefix):
                continue
            image = cv2.imread(str(self.settings.data_dir / url[len(prefix) :]))
            if image is not None:
                return int(image.shape[1]), int(image.shape[0])
        return None

    @staticmethod
    def _usable(bbox: dict | None, frame_width: int, frame_height: int):
        """(foot column, foot row, pixel height) when the box holds a whole person, else None."""

        if not isinstance(bbox, dict):
            return None
        try:
            x = float(bbox["x"])
            y = float(bbox["y"])
            width = float(bbox["width"])
            height = float(bbox["height"])
        except (KeyError, TypeError, ValueError):
            return None
        if height <= 0 or width <= 0:
            return None
        if float(bbox.get("confidence") or 0.0) < _MIN_CONFIDENCE:
            return None
        # Clipped by an edge means the box stops before the person does, and its height is a
        # measure of the frame rather than of them.
        if (
            x <= _EDGE_MARGIN
            or y <= _EDGE_MARGIN
            or x + width >= frame_width - _EDGE_MARGIN
            or y + height >= frame_height - _EDGE_MARGIN
        ):
            return None
        return x + width / 2.0, y + height, height
