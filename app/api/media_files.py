from pathlib import Path

from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from app.services.storage import IMAGE_SUFFIXES, VIDEO_SUFFIXES


class MediaStaticFiles(StaticFiles):
    """Serve only passive media, including files uploaded by older releases."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        if Path(path).suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            raise HTTPException(status_code=404)
        response = await super().get_response(path, scope)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'none'; sandbox"
        return response
