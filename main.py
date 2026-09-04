import base64
import binascii
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router, openai_compatible_router
from app.api.scene_summary import router as scene_summary_router
from app.config.settings import Settings, get_settings
from app.db.session import SessionLocal, init_db
from app.models.media import VideoStream
from app.services.storage import StorageService
from app.services.stream_runtime import stream_runtime
from app.services.vector_index_queue import vector_index_queue

logger = logging.getLogger(__name__)

NO_CACHE_HEADERS = {"Cache-Control": "no-store"}
# Built by `npm run build` in frontend/ at deploy time; not committed.
FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"
AUTH_EXEMPT_PATHS = {
    "/health",
    "/v1/embeddings",
    "/api/embeddings/visual",
    "/api/embeddings/image-vector",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    StorageService(settings).ensure_dirs()
    if settings.auto_create_tables:
        init_db()
    vector_index_queue.start(settings)
    if settings.stream_autostart_running:
        _autostart_running_streams()
    try:
        yield
    finally:
        vector_index_queue.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    StorageService(settings).ensure_dirs()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if settings.app_basic_auth_username and settings.app_basic_auth_password:
        app.middleware("http")(_basic_auth_middleware(settings))
    app.include_router(api_router)
    app.include_router(openai_compatible_router)
    app.include_router(scene_summary_router)
    app.mount("/data", StaticFiles(directory=settings.data_dir), name="data")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built Vue SPA, or explain how to build it if the bundle is missing.

    Hashed assets under /assets are immutable and cached hard; index.html is not, so a redeploy
    is picked up on the next load. Every other unmatched GET falls back to index.html because
    the router runs in history mode -- without that, opening /faces directly would 404.
    """

    index_file = FRONTEND_DIST / "index.html"
    if not index_file.is_file():
        logger.warning(
            "frontend bundle missing at %s; run `npm ci && npm run build` in frontend/",
            FRONTEND_DIST,
        )

        @app.get("/{_path:path}", include_in_schema=False)
        def frontend_missing(_path: str) -> Response:
            return Response(
                status_code=503,
                media_type="text/plain; charset=utf-8",
                content=(
                    "前端未构建。请在 frontend/ 执行:\n  npm ci && npm run build\n"
                    f"期望产物: {index_file}\n"
                ),
            )

        return

    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{spa_path:path}", include_in_schema=False)
    def spa(spa_path: str) -> FileResponse:
        # A real file under dist (favicon, robots.txt) wins; anything else is a client route.
        candidate = (FRONTEND_DIST / spa_path).resolve()
        if spa_path and candidate.is_file() and candidate.is_relative_to(FRONTEND_DIST.resolve()):
            return FileResponse(candidate)
        return FileResponse(index_file, headers=NO_CACHE_HEADERS)


def _basic_auth_middleware(settings: Settings):
    async def middleware(request: Request, call_next):
        if request.url.path in AUTH_EXEMPT_PATHS:
            return await call_next(request)
        if _is_basic_auth_valid(request.headers.get("authorization"), settings):
            return await call_next(request)
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="SightIndex"'},
            content="Authentication required",
        )

    return middleware


def _is_basic_auth_valid(authorization: str | None, settings: Settings) -> bool:
    if not authorization:
        return False
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "basic" or not token:
        return False
    try:
        decoded = base64.b64decode(token, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    username, separator, password = decoded.partition(":")
    if not separator:
        return False
    expected_username = settings.app_basic_auth_username or ""
    expected_password = settings.app_basic_auth_password or ""
    return secrets.compare_digest(username, expected_username) and secrets.compare_digest(
        password,
        expected_password,
    )


def _autostart_running_streams() -> int:
    started_count = 0
    with SessionLocal() as db:
        streams = (
            db.query(VideoStream)
            .filter(VideoStream.status.in_(["running", "starting"]))
            .all()
        )
    for stream in streams:
        if stream_runtime.start(stream.id) == "started":
            started_count += 1
    return started_count


app = create_app()
