"""Decode supported upload formats without third-party Image.open hooks."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import BinaryIO

from PIL import BmpImagePlugin, GifImagePlugin, Image, JpegImagePlugin, PngImagePlugin


@contextmanager
def open_supported_image(source: BinaryIO, *, max_pixels: int) -> Iterator[Image.Image]:
    """Use Pillow's codec classes on a seekable, byte-limited stream owned by the caller.

    Ultralytics replaces PIL.Image.open process-wide and retries failures with a HEIF
    plugin, including an attempted package install. Validation must not depend on import
    order or invoke that fallback. The signature selects a codec, whose parser validates
    the content. Check pixels before load: direct codecs bypass Image.open's bomb check.
    """
    source.seek(0)
    prefix = source.read(16)
    source.seek(0)
    if prefix.startswith(b"\xff\xd8\xff"):
        decoder = JpegImagePlugin.JpegImageFile
    elif prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        decoder = PngImagePlugin.PngImageFile
    elif prefix.startswith((b"GIF87a", b"GIF89a")):
        decoder = GifImagePlugin.GifImageFile
    elif prefix.startswith(b"BM"):
        decoder = BmpImagePlugin.BmpImageFile
    elif prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP":
        from PIL import WebPImagePlugin

        decoder = WebPImagePlugin.WebPImageFile
    else:
        raise Image.UnidentifiedImageError("File is not a supported image")
    with decoder(source) as image:
        if image.width * image.height > max_pixels:
            raise Image.DecompressionBombError("Image exceeds the pixel limit")
        image.load()
        yield image
