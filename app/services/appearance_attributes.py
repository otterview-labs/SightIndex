"""Clothing tone from the pixels, without a model download and without guessing.

The structured attribute pipeline needs a VLM and this deployment has no reachable one. Colour
is the one attribute readable straight from the image, so this fills in what the pixels support
and stays quiet about the rest.

Staying quiet is the design, and the measurements forced it. Over 300 crops from these cameras
the median saturation of a person's core is 29 and only 3% reach 80. Naming a hue at 29 produces
confident nonsense, and an index full of that is worse than an empty one: "紫色裤子" would return
a crowd of grey ones.

Two failures from a visual audit of the first version shaped the rest:

* A man in a black shirt and black trousers came back as blue, saturation 82. Near-black pixels
  have unstable hue and saturation -- there is barely any signal left to divide -- so brightness
  has to be settled before saturation is consulted, not after.
* Fixed horizontal bands sampled the wall above one person and the floor beside another. The
  crop is not tight around the body and the body is not centred in it, so the bands were
  measuring the corridor. Keypoints put the samples on the clothing, and when there are no
  keypoints the honest answer is nothing at all.

Three more attributes come off the same pose keypoints, each kept or dropped on a visual
audit rather than on whether it produced a value:

* Facing. Keypoint *presence* is not visibility -- YOLO-pose invents a nose for someone walking
  away, and a first attempt built on counting face keypoints called half the back views "front".
  The shoulders decide it instead: they are named for the person's own left and right, so the
  left shoulder sits on the image's right when the body faces the camera and the order flips
  when they turn. Checked against a balanced sheet, that got 12 of 12 backs.
* Trouser length, from skin showing on the shin. Sampled 40% down from the knee, because the
  knee-ankle midpoint kept landing on shoes.
* Sleeve length, which only ever claims "short". Skin on the forearm is evidence of a bare arm;
  no skin is not evidence of a sleeve, because a hanging hand, a folded arm and a forearm
  crossing the chest all sample clothing instead. Three attempts at guarding the sample point
  left "long" wrong nearly every time and "short" right nearly every time, so only the half
  that holds is published.

Posture and build were dropped. Posture measured horizontal foot separation, which is a
projection of walking *direction*: someone striding straight at an overhead camera has their
feet in line and read as standing. Build measured shoulder span over torso length, which
collapses as soon as the person turns.

Gender, age, hats, bags and behaviour are not attempted. Those need a model.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None  # type: ignore[assignment]

# OpenCV hue is degrees halved, so these bounds run 0..180. Ordered by upper bound.
_HUE_NAMES: tuple[tuple[int, str], ...] = (
    (8, "red"),
    (22, "orange"),
    (33, "yellow"),
    (78, "green"),
    (99, "cyan"),
    (129, "blue"),
    (155, "purple"),
    (180, "red"),
)

# COCO keypoint order from YOLO-pose.
_NOSE, _EARS = 0, (3, 4)
_L_SHOULDER, _R_SHOULDER = 5, 6
_SHOULDERS, _HIPS, _KNEES = (5, 6), (11, 12), (13, 14)
_ELBOWS, _WRISTS, _ANKLES = (7, 8), (9, 10), (15, 16)
_KEYPOINT_CONFIDENCE = 0.5


@dataclass(frozen=True)
class BodyShape:
    """What the skeleton supports saying. Every field may be None, and usually some are."""

    facing: str | None = None
    upper_length: str | None = None
    lower_length: str | None = None


@dataclass(frozen=True)
class ClothingTone:
    upper: str
    lower: str
    upper_saturation: float
    lower_saturation: float


@lru_cache(maxsize=1)
def _pose_model(weights: str):
    """Loads YOLO-pose once per process, or returns None if it cannot be had.

    Never downloads: the weights participate in the pinned inference manifest, and a service
    that fetches them on first use has already caused one eight-hour request here.
    """

    if not Path(weights).is_file():
        return None
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return YOLO(weights)
    except Exception:
        return None


class AppearanceAttributeService:
    """Reads clothing tone off a crop. No network, no weights beyond the pinned pose model."""

    def __init__(
        self,
        *,
        saturation_floor: float = 110.0,
        hue_value_floor: float = 90.0,
        dark_ratio: float = 0.90,
        pose_weights: str | None = None,
        sharpness_floor: float = 158.0,
    ) -> None:
        self.saturation_floor = saturation_floor
        self.hue_value_floor = hue_value_floor
        self.dark_ratio = dark_ratio
        self.pose_weights = pose_weights or str(Path.home() / ".cache" / "yolov8n-pose.pt")
        # Skin and cloth blur into each other on a smeared crop, and both wrong length readings
        # in the audit were motion blur. The floor is the 10th percentile of Laplacian variance
        # over 400 crops from these cameras, so it refuses the blurriest tenth and no more.
        self.sharpness_floor = sharpness_floor

    def describe(self, crop_path: Path) -> dict[str, object] | None:
        """Attributes in the shape the observation index already reads, or None if unreadable."""

        read = self._read(crop_path)
        if read is None:
            return None
        tone, shape = read
        clothing: dict[str, object] = {"upper_color": tone.upper, "lower_color": tone.lower}
        # Absent keys, not null ones: a search filtering on sleeve length should pass over a crop
        # that has no reading, not match it against a stated blank.
        if shape.upper_length:
            clothing["upper_length"] = shape.upper_length
        if shape.lower_length:
            clothing["lower_length"] = shape.lower_length
        attributes: dict[str, object] = {
            "object_type": "person",
            "clothing": clothing,
            # Named so a later VLM pass is recognisable as the better source and can replace it.
            "source": "cv_tone",
            "notes": "按关键点取样；饱和度不足时只给深浅，无把握的项留空",
        }
        if shape.facing:
            attributes["facing"] = shape.facing
        return attributes

    def tone(self, crop_path: Path) -> ClothingTone | None:
        read = self._read(crop_path)
        return None if read is None else read[0]

    def shape(self, crop_path: Path) -> BodyShape | None:
        read = self._read(crop_path)
        return None if read is None else read[1]

    def _read(self, crop_path: Path) -> tuple[ClothingTone, BodyShape] | None:
        """Loads the crop and runs the pose model once for both readings."""

        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - opencv ships with the vision extra
            return None
        if np is None:
            return None
        image = cv2.imread(str(crop_path))
        if image is None or image.size == 0:
            return None

        pose = self._pose(image)
        samples = self._keypoint_samples(cv2, image, pose)
        if samples is None:
            # No pose, no idea which pixels are clothing. Saying nothing beats measuring the wall.
            return None
        upper, lower = samples
        # Light and dark only mean anything against the light that was falling at the time. A
        # white shirt in shadow is darker than a black one in sun, and an absolute cut called
        # white tees dark for exactly that reason. The crop's own median is the local exposure.
        reference = float(np.median(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]))
        tone = ClothingTone(
            upper=self._name(upper, reference),
            lower=self._name(lower, reference),
            upper_saturation=float(upper[1]),
            lower_saturation=float(lower[1]),
        )
        return tone, self._shape(cv2, image, pose)

    def _pose(self, image):
        """(keypoints, confidences) for the largest body, or None."""

        model = _pose_model(self.pose_weights)
        if model is None:
            return None
        try:
            result = model(image, verbose=False)[0]
        except Exception:
            return None
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or len(keypoints.xy) == 0:
            return None
        points = keypoints.xy[0].cpu().numpy()
        if len(points) <= max(_ANKLES):
            return None
        scores = keypoints.conf
        confidence = (
            scores[0].cpu().numpy() if scores is not None else np.ones(len(points), dtype="float64")
        )
        return points, confidence

    def _keypoint_samples(self, cv2_module, image, pose=None):
        if pose is None:
            return None
        points, _ = pose
        # A patch scaled to the body, not the frame: a distant person needs a small one.
        radius = max(3, int(image.shape[1] * 0.08))
        upper = self._patch(cv2_module, image, points, _SHOULDERS, _HIPS, radius)
        lower = self._patch(cv2_module, image, points, _HIPS, _KNEES, radius)
        if upper is None or lower is None:
            return None
        return upper, lower

    def _shape(self, cv2_module, image, pose) -> BodyShape:
        if pose is None:
            return BodyShape()
        points, confidence = pose
        seen = lambda index: (  # noqa: E731 - a predicate, not a function worth a def
            confidence[index] >= _KEYPOINT_CONFIDENCE and bool(points[index].any())
        )
        facing = self._facing(points, seen)
        if not self._sharp_enough(cv2_module, image):
            # Length needs a skin/cloth edge, which blur erases. Facing needs only two positions
            # and survives it, so only the lengths are withheld.
            return BodyShape(facing=facing)
        skin = self._skin_reference(cv2_module, image, points, seen)
        if skin is None:
            return BodyShape(facing=facing)
        return BodyShape(
            facing=facing,
            upper_length=self._sleeve(cv2_module, image, points, seen, skin),
            lower_length=self._trouser(cv2_module, image, points, seen, skin),
        )

    @staticmethod
    def _facing(points, seen) -> str | None:
        if not (seen(_L_SHOULDER) and seen(_R_SHOULDER)):
            return None
        left, right = points[_L_SHOULDER], points[_R_SHOULDER]
        span = abs(float(left[0] - right[0]))
        separation = float(np.linalg.norm(left - right))
        if separation <= 0 or span / separation < 0.35:
            # Shoulders nearly stacked: seen edge-on, or from so far overhead that the projection
            # says nothing. Neither is a direction worth publishing.
            return None
        return "front" if left[0] > right[0] else "back"

    def _sharp_enough(self, cv2_module, image) -> bool:
        grey = cv2_module.cvtColor(image, cv2_module.COLOR_BGR2GRAY)
        return float(cv2_module.Laplacian(grey, cv2_module.CV_64F).var()) >= self.sharpness_floor

    def _skin_reference(self, cv2_module, image, points, seen):
        """This person's own face tone. Skin colour varies far too much to hard-code."""

        face = [points[index] for index in (_NOSE, *_EARS) if seen(index)]
        if not face:
            return None
        radius = max(2, int(image.shape[1] * 0.04))
        return self._window(cv2_module, image, np.mean(face, axis=0), radius)

    def _sleeve(self, cv2_module, image, points, seen, skin) -> str | None:
        """Return "short" for forearm skin; otherwise abstain instead of guessing "long"."""

        radius = max(3, int(image.shape[1] * 0.07))
        torso = self._torso_box(points, seen)
        for shoulder, elbow, wrist in zip(_SHOULDERS, _ELBOWS, _WRISTS, strict=True):
            if not (seen(elbow) and seen(wrist)):
                continue
            upper_arm = (
                float(np.linalg.norm(points[shoulder] - points[elbow]))
                if seen(shoulder)
                else 0.0
            )
            forearm = float(np.linalg.norm(points[elbow] - points[wrist]))
            if forearm < max(2.0 * radius, 0.6 * upper_arm):
                continue  # foreshortened: the patch would cover the whole arm and its background
            point = points[elbow] + (points[wrist] - points[elbow]) * 0.45
            if torso is not None and self._inside(point, torso):
                continue
            if self._is_skin(self._window(cv2_module, image, point, radius), skin):
                return "short"
        return None

    def _trouser(self, cv2_module, image, points, seen, skin) -> str | None:
        radius = max(3, int(image.shape[1] * 0.07))
        for knee, ankle in zip(_KNEES, _ANKLES, strict=True):
            if not (seen(knee) and seen(ankle)):
                continue
            if float(np.linalg.norm(points[knee] - points[ankle])) < 2.0 * radius:
                continue
            # 40% down the shin: the midpoint drifted onto shoes and socks, which are not trousers.
            point = points[knee] + (points[ankle] - points[knee]) * 0.40
            verdict = self._is_skin(self._window(cv2_module, image, point, radius), skin)
            if verdict is None:
                continue
            return "short" if verdict else "long"
        return None

    @staticmethod
    def _torso_box(points, seen):
        """Shoulders to hips: the shirt, and nothing more.

        An earlier version reached to the knees and out past the shoulders, to stop a hand
        resting on a thigh from being read as a sleeve. That guard existed to protect the "long"
        verdict, which is no longer published, and it cost most of the readings -- a forearm
        hanging at the side sits right at that widened edge, so sleeve length fired on 7% of
        crops. What is left to guard against is a skin-toned shirt read as a bare arm, and the
        garment's own outline is the right fence for that.
        """

        corners = [points[index] for index in (*_SHOULDERS, *_HIPS) if seen(index)]
        if len(corners) < 4:
            return None
        corners = np.array(corners)
        return (
            float(corners[:, 0].min()),
            float(corners[:, 0].max()),
            float(corners[:, 1].min()),
            float(corners[:, 1].max()),
        )

    @staticmethod
    def _inside(point, box) -> bool:
        left, right, top, bottom = box
        return left <= float(point[0]) <= right and top <= float(point[1]) <= bottom

    @staticmethod
    def _window(cv2_module, image, point, radius):
        x, y = int(point[0]), int(point[1])
        window = image[max(0, y - radius) : y + radius, max(0, x - radius) : x + radius]
        if window.size == 0:
            return None
        return np.median(
            cv2_module.cvtColor(window, cv2_module.COLOR_BGR2HSV).reshape(-1, 3), axis=0
        )

    @staticmethod
    def _is_skin(sample, reference) -> bool | None:
        """Compared on hue and saturation only: the same arm is lit unevenly along its length."""

        if sample is None or reference is None:
            return None
        gap = abs(float(sample[0]) - float(reference[0]))
        hue_gap = min(gap, 180.0 - gap)  # hue wraps, and skin sits near the wrap
        return hue_gap < 12.0 and abs(float(sample[1]) - float(reference[1])) < 45.0

    @staticmethod
    def _patch(cv2_module, image, points, first, second, radius):
        """Median HSV midway between two keypoint pairs -- the middle of a garment."""

        centres = []
        for pair in (first, second):
            left, right = points[pair[0]], points[pair[1]]
            if not left.any() or not right.any():
                continue
            centres.append((left + right) / 2)
        if not centres:
            return None
        windows = []
        for x, y in centres:
            x, y = int(x), int(y)
            window = image[max(0, y - radius) : y + radius, max(0, x - radius) : x + radius]
            if window.size:
                windows.append(cv2_module.cvtColor(window, cv2_module.COLOR_BGR2HSV).reshape(-1, 3))
        if not windows:
            return None
        return np.median(np.concatenate(windows), axis=0)

    def _name(self, hsv, reference: float) -> str:
        hue, saturation, value = float(hsv[0]), float(hsv[1]), float(hsv[2])
        # A hue needs saturation *and* light to stand on. Measured over 228 keypoint patches from
        # these cameras, saturation above 80 only ever occurs below value 100 -- the strong
        # colours are all shadow, where hue is noise. That is where the purple trousers and the
        # blue black-shirt came from. The floor sits at the 90th percentile so a hue is claimed
        # only for clothing that is genuinely colourful, which this footage rarely contains and
        # a better-lit camera might.
        if saturation >= self.saturation_floor and value >= self.hue_value_floor:
            for bound, name in _HUE_NAMES:
                if hue < bound:
                    return name
            return "red"
        # Not "grey": grey is a claim about colour, and there is no colour here to claim.
        # Measured against this crop's own exposure, so the answer survives a shaded doorway.
        return "dark" if value < reference * self.dark_ratio else "light"
