from io import BytesIO

import pytest
from PIL import Image

from app.services.image_decode import open_supported_image


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP", "BMP", "GIF"])
def test_upload_decoder_does_not_use_global_image_open(monkeypatch, image_format):
    content = BytesIO()
    Image.new("RGB", (16, 32), (80, 90, 100)).save(content, format=image_format)

    def forbidden(*args, **kwargs):
        pytest.fail("A global Image.open hook was invoked during upload validation")

    monkeypatch.setattr(Image, "open", forbidden)
    from app.services.image_decode import open_supported_image as decode_after_patch

    with decode_after_patch(content, max_pixels=512) as image:
        assert image.size == (16, 32)
    with pytest.raises(Image.DecompressionBombError):
        with open_supported_image(BytesIO(content.getvalue()), max_pixels=511):
            pass


@pytest.mark.parametrize("content", [b"", b"not an image", b"\x89PNG\r\n\x1a\nbroken"])
def test_invalid_content_never_reaches_global_fallback(monkeypatch, content):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid content reached the third-party Image.open fallback")

    monkeypatch.setattr(Image, "open", forbidden)
    with pytest.raises((Image.UnidentifiedImageError, SyntaxError, OSError)):
        with open_supported_image(BytesIO(content), max_pixels=1000):
            pass


def test_real_ultralytics_import_cannot_change_upload_validation(monkeypatch):
    pytest.importorskip("ultralytics")
    from ultralytics.utils import checks

    def forbidden(*args, **kwargs):
        pytest.fail("Upload validation attempted a model dependency installation")

    monkeypatch.setattr(checks, "check_requirements", forbidden)
    with pytest.raises(Image.UnidentifiedImageError):
        with open_supported_image(BytesIO(b"invalid"), max_pixels=1000):
            pass
