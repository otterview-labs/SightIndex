import threading

import cv2
import numpy as np
import pytest

from app.config.settings import Settings
from app.services.stream_runtime import StreamRuntime


def blurry_frame() -> np.ndarray:
    return np.full((64, 64, 3), 128, dtype=np.uint8)


def sharp_frame() -> np.ndarray:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[::2, ::2] = 255
    frame[1::2, 1::2] = 255
    return frame


class FakeCapture:
    def __init__(self, frames: list[tuple[bool, np.ndarray | None]]) -> None:
        self._frames = list(frames)
        self.read_count = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_count += 1
        return self._frames.pop(0)


@pytest.fixture
def runtime() -> StreamRuntime:
    return StreamRuntime()


def test_a_burst_of_one_never_reads_again(runtime):
    capture = FakeCapture([(True, sharp_frame())])
    settings = Settings(_env_file=None, stream_sharpest_frame_burst_size=1)
    first = blurry_frame()
    result = runtime._sharpest_of_burst(capture, first, cv2, settings, threading.Event())
    assert result is first
    assert capture.read_count == 0


def test_the_sharpest_frame_in_the_burst_wins_even_read_last(runtime):
    capture = FakeCapture([(True, blurry_frame()), (True, sharp_frame())])
    settings = Settings(_env_file=None, stream_sharpest_frame_burst_size=3)
    result = runtime._sharpest_of_burst(
        capture, blurry_frame(), cv2, settings, threading.Event()
    )
    assert np.array_equal(result, sharp_frame())
    assert capture.read_count == 2


def test_a_failed_extra_read_keeps_the_best_frame_seen_so_far(runtime):
    capture = FakeCapture([(True, blurry_frame()), (False, None)])
    settings = Settings(_env_file=None, stream_sharpest_frame_burst_size=5)
    first = sharp_frame()
    result = runtime._sharpest_of_burst(capture, first, cv2, settings, threading.Event())
    assert np.array_equal(result, first)
    assert capture.read_count == 2


def test_a_stop_signal_ends_the_burst_early(runtime):
    capture = FakeCapture([(True, sharp_frame())])
    settings = Settings(_env_file=None, stream_sharpest_frame_burst_size=4)
    stop_event = threading.Event()
    stop_event.set()
    first = blurry_frame()
    result = runtime._sharpest_of_burst(capture, first, cv2, settings, stop_event)
    assert result is first
    assert capture.read_count == 0
