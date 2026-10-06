"""Native person pixels, not display enhancement or background texture, drive retention."""

import json

import pytest

from app.services.crop_quality import measure_person_roi

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def texture(width: int = 128, height: int = 256):
    """Mid-exposure stripes with clear edges and no clipped black/white pixels."""

    stripes = np.where((np.indices((height, width))[1] // 4) % 2, 192, 64).astype(np.uint8)
    return np.repeat(stripes[:, :, None], 3, axis=2)


def test_blur_lowers_native_person_quality() -> None:
    frame = texture()
    blurred = cv2.GaussianBlur(frame, (11, 11), 4)
    bbox = {"x": 0, "y": 0, "width": 128, "height": 256}
    sharp = measure_person_roi(frame, bbox)
    soft = measure_person_roi(blurred, bbox)

    assert sharp is not None and soft is not None
    assert sharp.sharpness > soft.sharpness
    assert 0 <= soft.score < sharp.score <= 1
    assert sharp.width == 128 and sharp.height == 256
    assert json.loads(json.dumps(sharp.metadata()))["version"] == "person-roi-quality-v1"


def test_sharp_background_does_not_improve_a_blurry_person() -> None:
    person = cv2.GaussianBlur(texture(60, 120), (11, 11), 4)
    plain = np.full((200, 200, 3), 128, dtype=np.uint8)
    patterned = texture(200, 200)
    plain[40:160, 70:130] = person
    patterned[40:160, 70:130] = person
    bbox = {"x": 70, "y": 40, "width": 60, "height": 120}

    assert measure_person_roi(plain, bbox) == measure_person_roi(patterned, bbox)


@pytest.mark.parametrize(
    "bbox",
    [
        {},
        {"x": None, "y": 0, "width": 40, "height": 40},
        {"x": float("nan"), "y": 0, "width": 40, "height": 40},
        {"x": 0, "y": 0, "width": float("inf"), "height": 40},
        {"x": 0, "y": 0, "width": -1, "height": 40},
        {"x": 500, "y": 500, "width": 40, "height": 40},
        {"x": 0, "y": 0, "width": 2, "height": 2},
    ],
)
def test_invalid_roi_is_unknown_not_a_zero_quality_measurement(bbox) -> None:
    assert measure_person_roi(texture(), bbox) is None


@pytest.mark.parametrize(
    "frame",
    [
        object(),
        np.zeros((20,), dtype=np.uint8),
        np.zeros((20, 20, 2), dtype=np.uint8),
        np.zeros((20, 20), dtype=np.float32),
    ],
)
def test_unsupported_frame_is_unknown(frame) -> None:
    assert measure_person_roi(frame, {"x": 0, "y": 0, "width": 10, "height": 10}) is None


@pytest.mark.parametrize("channels", [None, 1, 3, 4])
def test_gray_and_color_frames_and_clipped_bounds(channels) -> None:
    gray = texture(32, 64)[:, :, 0]
    frame = gray if channels is None else np.repeat(gray[:, :, None], channels, axis=2)
    quality = measure_person_roi(frame, {"x": -4, "y": -3, "width": 40, "height": 70})

    assert quality is not None
    assert (quality.width, quality.height) == (32, 64)
    assert 0 <= quality.score <= 1


def test_large_roi_is_bounded_without_losing_native_dimensions() -> None:
    quality = measure_person_roi(texture(512, 1024), {"x": 0, "y": 0, "width": 512, "height": 1024})

    assert quality is not None
    assert (quality.width, quality.height) == (512, 1024)


def test_clipped_exposure_cannot_win_on_high_contrast_edges() -> None:
    pixels = texture()
    pixels[pixels == 64] = 0
    pixels[pixels == 192] = 255
    quality = measure_person_roi(pixels, {"x": 0, "y": 0, "width": 128, "height": 256})

    assert quality is not None
    assert quality.sharpness > 0
    assert quality.usable_exposure == 0
    assert quality.score == 0


def test_missing_vision_is_unknown(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "cv2", None)
    assert measure_person_roi(texture(), {"x": 0, "y": 0, "width": 128, "height": 256}) is None
