"""Measure native person pixels for bounded, per-visit frame retention.

These are image-quality hints, not identity, face-quality or whole-body confidence. They never
compare two people's identities and do not reject a crop by themselves. Measurements use the
un-padded, un-enhanced ROI, so a textured background or display sharpening cannot win a frame.
"""

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PersonRoiQuality:
    """Native-pixel measurements with a bounded retention score, not a probability."""

    width: int
    height: int
    sharpness: float
    brightness: float
    usable_exposure: float
    score: float
    version: str = "person-roi-quality-v1"

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible measurements for later input-quality evaluation."""

        return asdict(self)


def measure_person_roi(frame: object, bbox: Mapping[str, Any]) -> PersonRoiQuality | None:
    """Measure a valid uint8 ROI; unavailable vision or invalid pixels remain unknown.

    Args:
        frame: The original decoded BGR or grayscale frame, before JPEG/display enhancement.
        bbox: Person bounding box in native pixels, with x, y, width and height.

    Returns:
        Quality hints, or None if the ROI cannot safely be measured. Unknown quality must not
        be mistaken for a blurry image or used to discard a previously usable detection.
    """

    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    try:
        pixels = np.asarray(frame)
        if pixels.dtype != np.uint8 or pixels.ndim not in (2, 3):
            return None
        if pixels.ndim == 3 and pixels.shape[2] not in (1, 3, 4):
            return None
        frame_height, frame_width = pixels.shape[:2]
        x, y, width, height = (float(bbox[key]) for key in ("x", "y", "width", "height"))
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            return None
        if width <= 0 or height <= 0:
            return None
        x1 = max(0, min(frame_width, math.floor(x)))
        y1 = max(0, min(frame_height, math.floor(y)))
        x2 = max(0, min(frame_width, math.ceil(x + width)))
        y2 = max(0, min(frame_height, math.ceil(y + height)))
        roi_width, roi_height = x2 - x1, y2 - y1
        if roi_width < 3 or roi_height < 3:
            return None
        roi = pixels[y1:y2, x1:x2]
        if pixels.ndim == 2:
            gray = roi
        elif roi.shape[2] == 1:
            gray = roi[:, :, 0]
        else:
            gray = cv2.cvtColor(
                roi, cv2.COLOR_BGRA2GRAY if roi.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            )
        # Bound CPU work; never upsample a small person and invent apparent native detail.
        scale = min(1.0, 256.0 / max(roi_width, roi_height))
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (max(3, round(roi_width * scale)), max(3, round(roi_height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        usable_exposure = float(np.mean((gray > 8) & (gray < 247)))
        resolution_weight = min(1.0, math.sqrt(roi_width * roi_height / (128.0 * 256.0)))
        # This heuristic chooses improvements within one visit; it is not a calibrated gate.
        score = resolution_weight * usable_exposure * sharpness / (sharpness + 100.0)
        if not all(
            math.isfinite(value) for value in (sharpness, brightness, usable_exposure, score)
        ):
            return None
        return PersonRoiQuality(
            width=roi_width,
            height=roi_height,
            sharpness=round(sharpness, 4),
            brightness=round(brightness, 4),
            usable_exposure=round(usable_exposure, 4),
            score=round(score, 6),
        )
    except (KeyError, TypeError, ValueError, OverflowError, cv2.error):
        return None
