"""An opt-in, sample-only public console; never expose the ordinary workspace anonymously."""

import asyncio
import re
from collections import deque
from time import monotonic

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config.settings import Settings

UUID = r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
READ_PATHS = re.compile(
    rf"^(?:/health|/|/search|/observations|/api/(?:streams|images|person-crops|persons"
    rf"|media/counts|search/observations|search/semantic/status)"
    rf"|/api/(?:images|person-crops)/{UUID})$"
)
SEARCH_PATH = "/api/search/person-crops"


def validate_public_demo(settings: Settings) -> None:
    """Fail closed if an operator accidentally gives the demo active model/camera services."""
    disabled_flags = (
        "milvus_enabled", "reid_enabled", "vector_index_on_ingest",
        "vlm_structured_on_ingest", "vlm_structured_background", "face_recognition_on_ingest",
        "stream_autostart_running", "face_insightface_allow_download",
    )
    if any(getattr(settings, name) for name in disabled_flags):
        raise ValueError("Public demo requires cameras, recognition, model downloads and jobs off")
    if any(getattr(settings, name) != "none" for name in (
        "embedding_provider", "visual_embedding_provider", "vlm_provider",
    )):
        raise ValueError("Public demo requires all external model providers off")
    if not settings.database_url.startswith("sqlite:"):
        raise ValueError("Public demo must use its own disposable SQLite database")


class PublicDemoGuard:
    """Default-deny routes and bound the only permitted POST before FastAPI parses its body.

    No client IP headers are trusted. This small sample service uses a global request budget,
    a fixed-size time window and a concurrency cap, independent of proxy address attribution.
    """

    def __init__(self, app: ASGIApp, requests_per_minute: int = 600) -> None:
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.recent: deque[float] = deque()
        self.in_flight = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            await self.app(scope, receive, send)
            return
        method, path = scope["method"], scope["path"]
        is_static = path.startswith(("/assets/", "/data/"))
        allowed = method in ("GET", "HEAD") and (is_static or READ_PATHS.fullmatch(path))
        is_search = method == "POST" and path == SEARCH_PATH
        if not allowed and not is_search:
            await JSONResponse(
                {"detail": "Public demo is read-only. Self-host to upload or edit."},
                status_code=403,
            )(scope, receive, send)
            return
        if len(scope.get("query_string", b"")) > 2048:
            await JSONResponse({"detail": "Query is too long"}, status_code=414)(
                scope, receive, send,
            )
            return
        # Assets and health do not consume the API/HTML budget. All data is bundled sample media.
        if not is_static and path != "/health":
            now = monotonic()
            while self.recent and self.recent[0] < now - 60:
                self.recent.popleft()
            if len(self.recent) >= self.requests_per_minute or self.in_flight >= 8:
                await JSONResponse(
                    {"detail": "Demo is busy; please retry in a minute"}, status_code=429,
                    headers={"Retry-After": "60"},
                )(scope, receive, send)
                return
            self.recent.append(now)
        if is_search:
            body = bytearray()
            try:
                async with asyncio.timeout(10):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        body.extend(message.get("body", b""))
                        if len(body) > 4096:
                            await JSONResponse(
                                {"detail": "Demo search body exceeds 4 KB"}, status_code=413,
                            )(scope, receive, send)
                            return
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                await JSONResponse({"detail": "Request timed out"}, status_code=408)(
                    scope, receive, send,
                )
                return
            original_receive = receive
            replayed = False

            async def replay() -> dict:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await original_receive()

            receive = replay
        self.in_flight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.in_flight -= 1
