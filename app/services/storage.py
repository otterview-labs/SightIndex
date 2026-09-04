import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config.settings import Settings


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
        self.ensure_dirs()
        suffix = Path(file.filename or "").suffix or ".bin"
        filename = f"{uuid.uuid4()}{suffix}"
        target = self.settings.uploads_dir / filename
        with target.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        return f"/data/uploads/{filename}"

    def save_video(self, file: UploadFile) -> str:
        self.ensure_dirs()
        suffix = Path(file.filename or "").suffix or ".mp4"
        filename = f"{uuid.uuid4()}{suffix}"
        target = self.settings.videos_dir / filename
        with target.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        return f"/data/videos/{filename}"

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
