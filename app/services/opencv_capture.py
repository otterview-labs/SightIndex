import os
from typing import Any

from app.config.settings import Settings


def open_video_capture(cv2: Any, source: str, settings: Settings) -> Any:
    if not str(source).lower().startswith("rtsp://"):
        return cv2.VideoCapture(source)

    _configure_rtsp_ffmpeg_options(settings)
    params = _capture_params(cv2, settings)
    api_preference = getattr(cv2, "CAP_FFMPEG", 0)

    if params:
        try:
            capture = cv2.VideoCapture(source, api_preference, params)
            if capture.isOpened():
                _apply_capture_properties(capture, cv2)
                return capture
            capture.release()
        except Exception:
            pass

    try:
        capture = cv2.VideoCapture(source, api_preference)
    except Exception:
        capture = cv2.VideoCapture(source)
    _apply_capture_properties(capture, cv2)
    return capture


def _configure_rtsp_ffmpeg_options(settings: Settings) -> None:
    options: list[str] = []
    transport = settings.rtsp_transport.strip().lower()
    if transport in {"tcp", "udp", "udp_multicast", "http", "https"}:
        options.extend(["rtsp_transport", transport])
    if not options:
        return

    option_text = ";".join(options)
    existing = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    if "rtsp_transport" in existing:
        return
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"{existing}|{option_text}" if existing else option_text
    )


def _capture_params(cv2: Any, settings: Settings) -> list[int]:
    params: list[int] = []
    open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    if open_timeout is not None:
        params.extend([int(open_timeout), int(settings.rtsp_open_timeout_ms)])
    read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if read_timeout is not None:
        params.extend([int(read_timeout), int(settings.rtsp_read_timeout_ms)])
    return params


def _apply_capture_properties(capture: Any, cv2: Any) -> None:
    buffer_size = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
    if buffer_size is None:
        return
    try:
        capture.set(buffer_size, 1)
    except Exception:
        return
