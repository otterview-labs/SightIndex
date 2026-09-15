from starlette.datastructures import Headers
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config.settings import Settings


class _UploadSizeExceeded(MultiPartException):
    pass


class UploadSizeLimitMiddleware:
    """Bound request bodies before multipart spooling or JSON decoding."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.image_limit = settings.upload_image_max_bytes
        self.video_limit = settings.upload_video_max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = (
            self.video_limit
            if scope["path"].rstrip("/") == "/api/videos/upload"
            else (self.image_limit * 4 + 2) // 3
        ) + 64 * 1024
        response = JSONResponse({"detail": "Request body exceeds the size limit"}, status_code=413)
        length = Headers(scope=scope).get("content-length")
        if length is not None:
            try:
                declared_length = int(length)
            except ValueError:
                declared_length = -1
            if declared_length < 0:
                await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                    scope, receive, send
                )
                return
            if declared_length > limit:
                await response(scope, receive, send)
                return
        received = 0
        exceeded = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _UploadSizeExceeded("Request body exceeds the size limit")
            return message

        async def limited_send(message: Message) -> None:
            nonlocal response_started
            if not exceeded:
                if message["type"] == "http.response.start":
                    response_started = True
                await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except Exception:
            if not exceeded or response_started:
                raise
        if exceeded:
            if response_started:
                raise RuntimeError("Request size limit exceeded after response started")
            await response(scope, receive, send)
