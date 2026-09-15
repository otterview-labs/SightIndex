import uuid
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import BinaryIO

from fastapi import UploadFile

from app.config.settings import Settings

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})
VIDEO_SUFFIXES = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".ts"}
)


class InvalidUploadError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class StorageService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def ensure_dirs(self) -> None:
        for path in (
            self.settings.uploads_dir,
            self.settings.videos_dir,
            self.settings.crops_dir,
            self.settings.thumbnails_dir,
            self.settings.frames_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def save_upload(self, file: UploadFile) -> str:
        """Validate and losslessly re-encode one image before publishing it."""
        from PIL import Image, ImageOps, UnidentifiedImageError

        self.ensure_dirs()
        suffix = Path(file.filename or "").suffix.lower()
        if suffix and suffix not in IMAGE_SUFFIXES:
            raise InvalidUploadError("Unsupported image file type", 415)
        limit = self.settings.upload_image_max_bytes
        with (
            SpooledTemporaryFile(max_size=1024 * 1024) as source,
            SpooledTemporaryFile(max_size=1024 * 1024) as encoded,
        ):
            self._copy_limited(file.file, source, limit)
            source.seek(0)
            try:
                with Image.open(source, formats=["JPEG", "PNG", "WEBP", "BMP", "GIF"]) as image:
                    if image.width * image.height > self.settings.upload_image_max_pixels:
                        raise InvalidUploadError("Image exceeds the pixel limit", 413)
                    image.load()
                    with ImageOps.exif_transpose(image) as oriented:
                        with oriented.convert("RGB") as pixels:
                            pixels.info.clear()
                            pixels.save(encoded, format="PNG")
            except Image.DecompressionBombError as exc:
                raise InvalidUploadError("Image exceeds the pixel limit", 413) from exc
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
                raise InvalidUploadError("File is not a valid supported image") from exc
            if encoded.tell() > limit:
                raise InvalidUploadError("Normalized image is too large", 413)
            encoded.seek(0)
            filename = f"{uuid.uuid4()}.png"
            self._save_limited(encoded, self.settings.uploads_dir / filename, limit)
        return f"/data/uploads/{filename}"

    def save_video(self, file: UploadFile) -> str:
        self.ensure_dirs()
        suffix = Path(file.filename or "").suffix.lower() or ".mp4"
        if suffix not in VIDEO_SUFFIXES:
            raise InvalidUploadError("Unsupported video file type", 415)
        filename = f"{uuid.uuid4()}{suffix}"
        target = self.settings.videos_dir / filename
        self._save_limited(file.file, target, self.settings.upload_video_max_bytes)
        return f"/data/videos/{filename}"

    @staticmethod
    def _copy_limited(source: BinaryIO, output: BinaryIO, limit: int) -> None:
        total = 0
        while chunk := source.read(min(64 * 1024, limit - total + 1)):
            total += len(chunk)
            if total > limit:
                raise InvalidUploadError("Upload exceeds the file size limit", 413)
            output.write(chunk)

    @classmethod
    def _save_limited(cls, source: BinaryIO, target: Path, limit: int) -> None:
        output = target.open("xb")
        try:
            with output:
                cls._copy_limited(source, output, limit)
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def remove_data_url(self, url: str) -> None:
        """Remove one internally generated /data URL without allowing path traversal."""

        prefix = "/data/"
        if not url.startswith(prefix):
            return
        root = self.settings.data_dir.resolve()
        target = (root / url.removeprefix(prefix)).resolve()
        if not target.is_relative_to(root):
            return
        try:
            target.unlink(missing_ok=True)
        except OSError:
            return
