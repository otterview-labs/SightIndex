"""A bounded RTSP decoder lifetime, including native calls that ignore timeouts."""

import multiprocessing
import threading
import time
from multiprocessing.connection import Connection
from typing import Any

from app.config.settings import Settings


def _decode(connection: Connection, source: str, options: dict[str, Any]) -> None:
    # A fresh process avoids sharing FFmpeg and CUDA state with the API's worker threads.
    import cv2

    from app.services.opencv_capture import open_video_capture

    capture = None
    try:
        capture = open_video_capture(cv2, source, Settings(_env_file=None, **options))
        connection.send(capture.isOpened())
        while capture.isOpened() and connection.recv() == "read":
            connection.send(capture.read())
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        if capture is not None:
            capture.release()
        connection.close()


class CaptureProcess:
    """Own exactly one decoder process; stop/reconnect reaps it before creating another."""

    def __init__(
        self,
        source: str,
        settings: Settings,
        stop_event: threading.Event,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._stop_event = stop_event
        self._read_timeout = settings.rtsp_read_timeout_ms / 1000.0
        self._opened = False
        self._closed = False
        options = {
            field: getattr(settings, field)
            for field in ("rtsp_transport", "rtsp_open_timeout_ms", "rtsp_read_timeout_ms")
        }
        self._process = context.Process(
            target=_decode,
            args=(child, source, options),
            daemon=True,
            name="sightindex-rtsp-decoder",
        )
        try:
            self._process.start()
            child.close()
            self._opened = bool(self._receive(settings.rtsp_open_timeout_ms / 1000.0 + 2))
        except BaseException:
            child.close()
            self.release()
            raise
        if not self._opened:
            self.release()

    def _receive(self, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Camera decoder timed out")
            if self._connection.poll(min(remaining, 0.1)):
                return self._connection.recv()
            if not self._process.is_alive():
                raise OSError("Camera decoder exited")
        raise InterruptedError("Camera capture stopped")

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV-compatible boundary
        return self._opened and not self._closed

    def read(self) -> tuple[bool, Any]:
        self._connection.send("read")
        return self._receive(self._read_timeout)

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._opened = False
        self._connection.close()
        process = self._process
        if process.pid is None:
            return
        # Never call VideoCapture.release concurrently with a blocked native read. Terminating
        # its isolated owner lets the OS reclaim decoder memory, sockets and threads together.
        if process.is_alive():
            process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        if process.is_alive():
            raise RuntimeError("Camera decoder could not be reaped; refusing to reconnect")
        process.close()
