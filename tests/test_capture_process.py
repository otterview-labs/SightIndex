"""Native decoder timeouts must bound resource lifetime, not just caller wait time."""

import multiprocessing
import threading
import time

import pytest


def _fake_decode(connection, source, options):
    if source == "stuck-open":
        time.sleep(60)
    if source == "crash":
        connection.close()
        return
    connection.send(True)
    while connection.recv() == "read":
        if source == "stuck-read":
            time.sleep(60)
        connection.send((True, "frame"))


def _capture(monkeypatch, source, stop=None):
    from app.config.settings import Settings
    from app.services import capture_process

    monkeypatch.setattr(capture_process, "_decode", _fake_decode)
    return capture_process.CaptureProcess(
        source,
        Settings(_env_file=None, rtsp_read_timeout_ms=100, rtsp_open_timeout_ms=100),
        stop or threading.Event(),
    )


def _decoder_pids():
    return {p.pid for p in multiprocessing.active_children() if p.name == "sightindex-rtsp-decoder"}


def test_capture_process_returns_frames_and_releases_idempotently(monkeypatch):
    capture = _capture(monkeypatch, "normal")
    pid = capture._process.pid
    try:
        assert capture.isOpened()
        assert capture.read() == (True, "frame")
    finally:
        capture.release()
        capture.release()
    assert not capture.isOpened()
    assert pid not in _decoder_pids()


def test_repeated_stuck_reads_reap_every_decoder_before_reconnect(monkeypatch):
    original = _decoder_pids()
    for _ in range(3):
        capture = _capture(monkeypatch, "stuck-read")
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError):
                capture.read()
        finally:
            capture.release()
        assert time.monotonic() - started < 4
        assert _decoder_pids() == original


@pytest.mark.parametrize("source, error", [("stuck-open", TimeoutError), ("crash", EOFError)])
def test_failed_open_reaps_child(monkeypatch, source, error):
    original = _decoder_pids()
    with pytest.raises(error):
        _capture(monkeypatch, source)
    assert _decoder_pids() == original


def test_stop_interrupts_inflight_read_and_reaps_child(monkeypatch):
    stop = threading.Event()
    capture = _capture(monkeypatch, "stuck-read", stop)
    capture._read_timeout = 30
    timer = threading.Timer(0.2, stop.set)
    timer.start()
    try:
        started = time.monotonic()
        with pytest.raises(InterruptedError):
            capture.read()
        assert time.monotonic() - started < 2
    finally:
        timer.join()
        capture.release()
