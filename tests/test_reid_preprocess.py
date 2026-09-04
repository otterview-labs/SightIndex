"""Deterministic checks on the SapiensID eval-time preprocessing.

The service must not stretch rectangular body crops square: upstream's test transform is
ToTensor -> SquarePad(fill=1) -> Resize -> Normalize, and these tests pin that behaviour by
calling the vendored upstream code itself. SapiensModel.make_test_transform only reads four
attributes, so it is invoked unbound against a namespace to avoid constructing the 412M net.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

VENDOR = Path(__file__).parent.parent / "deploy" / "agx" / "reid_service" / "sapiensid"
for entry in (VENDOR / "tasks" / "sapiensID", VENDOR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from PIL import Image  # noqa: E402
from src.models.sapiens_id import SapiensModel, SquarePad  # noqa: E402

INPUT_SIZE = (384, 384)


def make_transform(square_pad: bool = True):
    stub = SimpleNamespace(
        input_size=INPUT_SIZE,
        rgb_mean=[0.5, 0.5, 0.5],
        rgb_std=[0.5, 0.5, 0.5],
        square_pad=square_pad,
    )
    return SapiensModel.make_test_transform(stub)


def solid_image(width: int, height: int, rgb=(255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (width, height), rgb)


def test_square_pad_pads_symmetrically_with_white():
    tensor = torch.zeros(3, 100, 200)  # landscape: h=100, w=200
    padded = SquarePad(fill=1)(tensor)
    assert padded.shape == (3, 200, 200)
    # 100 rows of padding split 50 top / 50 bottom, filled with 1 (white in 0..1 space).
    assert torch.all(padded[:, :50, :] == 1)
    assert torch.all(padded[:, 150:, :] == 1)
    assert torch.all(padded[:, 50:150, :] == 0)


def test_square_pad_splits_odd_padding():
    tensor = torch.zeros(3, 3, 2)  # 1 column of padding to distribute
    padded = SquarePad(fill=1)(tensor)
    assert padded.shape == (3, 3, 3)
    # wp=1: left gets 0, right gets 1.
    assert torch.all(padded[:, :, 2] == 1)
    assert torch.all(padded[:, :, :2] == 0)


def test_landscape_crop_is_padded_not_stretched():
    transform = make_transform()
    out = transform(solid_image(200, 100))
    assert out.shape == (3, *INPUT_SIZE)
    # Content occupies the middle half vertically; above and below is white padding,
    # which normalises to (1 - 0.5) / 0.5 = 1.0.
    assert torch.all(torch.abs(out[:, :90, :] - (1.0)) < 1e-5)
    assert torch.all(torch.abs(out[:, 294:, :] - (1.0)) < 1e-5)
    # Red content: R channel (255 -> 1.0 -> 1.0), G/B (0 -> -1.0).
    middle = out[:, 192, :]
    assert torch.all(torch.abs(middle[0] - (1.0)) < 1e-5)
    assert torch.all(torch.abs(middle[1] - (-1.0)) < 1e-5)


def test_portrait_crop_is_padded_left_and_right():
    transform = make_transform()
    out = transform(solid_image(100, 200))
    assert out.shape == (3, *INPUT_SIZE)
    assert torch.all(torch.abs(out[:, :, :90] - (1.0)) < 1e-5)
    assert torch.all(torch.abs(out[:, :, 294:] - (1.0)) < 1e-5)
    assert torch.all(torch.abs(out[1, :, 192] - (-1.0)) < 1e-5)


def test_square_crop_gets_no_padding():
    transform = make_transform()
    out = transform(solid_image(150, 150, rgb=(0, 255, 0)))
    assert out.shape == (3, *INPUT_SIZE)
    # No padding anywhere: every pixel is the green content.
    assert torch.all(torch.abs(out[1] - (1.0)) < 1e-5)
    assert torch.all(torch.abs(out[0] - (-1.0)) < 1e-5)


def test_without_square_pad_the_image_is_stretched():
    """The control: square_pad=false is the stretching behaviour the service must not use."""

    transform = make_transform(square_pad=False)
    out = transform(solid_image(200, 100))
    assert out.shape == (3, *INPUT_SIZE)
    # Stretched: no white padding rows at all.
    assert torch.all(torch.abs(out[1, 0, :] - (-1.0)) < 1e-5)
